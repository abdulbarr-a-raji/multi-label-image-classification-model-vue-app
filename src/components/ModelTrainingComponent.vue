<template>
  <div class="training">
    <h2>{{ header }}</h2>
    <label for="json-data">enter your dataset in JSON format here...</label>
    <textarea name="" id="json-data" rows="10" cols="50" v-model="jsonInput"></textarea>
    <br>
    <input type="file" id="imagedataUpload" multiple accept="image/*" />
    <br>
    <br>
    <button id="train-with-extractor" v-on:click="fitClassifcationHead">
      Load Image Data & Train Classification Head
    </button>
    <br>
    <small>uses tf.js for ML</small>
    <div id="micro-out-div"></div>
  </div>
</template>

<script>
export default {
  name: 'ModelTraining',
  props: {
    header: String,
    w: {type: Number, required: true},
    h: {type: Number, required: true}
  },
  mounted() {
    this.tf = window.tf;
  },
  data() {
    return {
      tf: null,
      featureExtractor: null,
      fully_connected_head: null,
      jsonInput: `[
  {
    "file_name": "",
    "labels": [0, 1]
  }
]`,
      labelNames: ["strawberry", "orange", "kiwi"],

      // FLAGS & COUNTERS

    }
  },
  methods: {
    checkForTensorflow() {
      if (this.tf) {
        console.log("tf.js is in this scope :D");
      } else {
        console.error("tf.js is NOT in this scope D':");
      }
    },
    async loadTrainingData() {
      const inputElement = document.getElementById('imagedataUpload');
      const uploaded = Array.from(inputElement.files);
      const training_dataset = JSON.parse(this.jsonInput);

      console.log("File names:-\n");
      uploaded.forEach((file) => console.log(file.name));

      console.log("training data before:-\n", training_dataset);

      for (const item of training_dataset) {
        const file = uploaded.find((value) => value.name == item.file_name);
        const img = await this.readFileAsImage(file);
        item.tensor = this.preprocessImageData(img);
      }

      console.log("training data after:-\n", training_dataset);

      /* must try...catch for this error eventually:
        Uncaught (in promise) Error: 
        Pass at least one tensor to tf.stack
        at loadTrainingData (data.js)

        | it occurs when no images are uploaded, 
        | but button has been pressed
      */

      return training_dataset;
    },
    readFileAsImage(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
          const img = new Image();
          img.src = reader.result;

          img.onload = () => resolve(img);
          img.onerror = reject;
        };

        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    },
    preprocessImageData(img) {
      return this.tf.tidy(() => {
        const tns = this.tf.browser.fromPixels(img);
        
        // setting crop size operation...
        const widthToHeight = tns.shape[1] / tns.shape[0];
        let croppedSize;

        if (widthToHeight > 1) {
          // Image is wider than tall? crop sides
          const heightToWidth = tns.shape[0] / tns.shape[1];
          const cropTop = (1 - heightToWidth) / 2;
          const cropBottom = 1 - cropTop;
          croppedSize = [[cropTop, 0, cropBottom, 1]];
        } else {
          // Image is taller than wide? crop top & bottom
          const cropLeft = (1 - widthToHeight) / 2;
          const cropRight = 1 - cropLeft;
          croppedSize = [[0, cropLeft, 1, cropRight]];
        }

        // crop, resize, and more...
        const croppedImgTns = this.tf.image.cropAndResize(
          tns.expandDims(0),
          croppedSize,
          [0],
          [this.w, this.h]
        ).toFloat()
          .div(255.0);
        
        // disposal/s
        tns.dispose();

        return this.featureExtractor.predict(croppedImgTns).squeeze();
      });
    },
    async setupModels() {
      const NUM_CLASSES = this.labelNames.length;

      try {
        // loading the feature extractor model...
        const extractorURL =
          "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
        const loadedModel = await this.tf.loadLayersModel(extractorURL);

        // Creating the feature extractor base model...
        const beforeFinalLayer = loadedModel.getLayer(
          "global_average_pooling2d_1"
        );
        this.featureExtractor = this.tf.model({
          inputs: loadedModel.inputs,
          outputs: beforeFinalLayer.output,
        });

        // Warming up the feature extractor...
        this.tf.tidy(() => {
          const warmupInput = this.tf.zeros([1, this.w, this.h, 3]);
          const answer = this.featureExtractor.predict(warmupInput);
          answer.print();
        });

        console.log("MobileNet v2 feature extractor created successfully");
      } catch (error) {
        console.error(`Error setting up the feature extractor: ${error}`);
        throw error;
      }

      // define architecture of classification head
      this.fully_connected_head = this.tf.sequential();
      this.fully_connected_head.add(
        this.tf.layers.dense({
          inputShape: [this.featureExtractor.outputs[0].shape[1]],
          units: 64,
          activation: "relu",
        })
      );
      this.fully_connected_head.add(
        this.tf.layers.dense({
          units: 32,
          activation: "relu",
        })
      );
      this.fully_connected_head.add(
        this.tf.layers.dense({
          units: NUM_CLASSES,
          activation: "sigmoid",
        })
      );

      // Compile classification head
      const model_optimizer = this.tf.train.adam();
      this.fully_connected_head.compile({
        optimizer: model_optimizer,
        loss: "binaryCrossentropy",
        metrics: [this.tf.metrics.binaryAccuracy],
      });

      console.log("Model created successfully");
      this.fully_connected_head.summary();
    },
    async fitClassifcationHead() {
      if(!this.fully_connected_head || !this.featureExtractor) await this.setupModels();

      // preprocess image data and load as tensors
      const training_data = await this.loadTrainingData();

      const xs = this.tf.stack(training_data.map((item) => item.tensor)); // tensor shape: [batch, height, width, channels]
      const ys = this.tf.stack(training_data.map((item) => item.labels)); // tensor shape: [batch, labels]

      console.log(xs.shape);
      console.log(ys.shape);

        const ls = training_data.map((item) => item.labels);

      document.getElementById('micro-out-div').innerText = ls.map((item) => item.map((label, i) => label == 1 ? this.labelNames[i] : "_")).join("\n");

      const BATCH_SIZE = 3;
      const NUM_EPOCHS = 10;
      const start = performance.now();

      const results = await this.fully_connected_head.fit(xs, ys, {
        shuffle: true,
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS
      });

      const end = performance.now();
      const timeTakenInSeconds = ((end - start) / 1000).toFixed(2);
      console.log(`Training completed in ${timeTakenInSeconds} seconds`);

      // Clean up tensors
      xs.dispose();
      ys.dispose();

      console.log("Training results:", results.history);

      this.$emit('training-is-complete', true);

      // save the model
      await this.fully_connected_head.save("downloads://pretrained-head");
    }
  }
}
</script>

<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}

label:has(+ textarea),
textarea {
  display: block;

  font-size: 0.8rem;
  letter-spacing: 1px;
}

label:has(+ textarea) {
  margin-bottom: 10px;
}

textarea {
  padding: 10px;
  margin: auto;

  max-width: 100%;

  line-height: 1.5;
  border-radius: 5px;
  border: 1px solid #cccccc;
  box-shadow: 1px 1px 1px #999999;
}

button {
  padding: 8px;

  background-color: #CC2936;
  border: none;
  border-radius: 4px;

  color: white;
  font-size: 14px;
  font-weight: bold;
  letter-spacing: 1.25px;

  opacity: 1;
  transition: opacity 0.28s cubic-bezier(.4, 0, .2, 1);
}

button:hover {
  opacity: 0.85;
}
</style>
