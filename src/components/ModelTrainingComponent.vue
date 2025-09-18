<template>
  <div class="training">
    <h2>{{ header }}</h2>
    <nav>
      <button class="persian-green tab" @click="current_tab = 1">View the Dataset</button>
      <button class="saffron tab" @click="current_tab = 2">Train the Model</button>
      <button class="sandy-brown tab" @click="current_tab = 3">Test the Model</button>
      <button class="sandy-brown tab" @click="exportModel">Save your Model</button>
    </nav>
    <main>
      <section ref="dataset" class="plain-container" v-show="current_tab == 1">
        <div v-for="(img, i) in training_dataset" :key="i" ref="image-card" class="image-card">
          <img :src="`/assets/training-images/${img.filename}`"/>
          <p>{{ img.labels.map((label, i) => label ? objects[i] : "").filter((s) => s !== "").join(" - ") || "none" }}</p>
        </div>
      </section>
      <section class="plain-container" v-show="current_tab == 2">
        <progress :value="trainingProgress" max="100"></progress>
        <p>Epoch {{ currentEpoch }} <br> Loss: {{ loss }} <br> Acc: {{ acc }}</p>
        <button id="train-with-extractor" v-on:click="fitClassifcationHead">
          Train the Classification Head
        </button>
      </section>
      <section class="plain-container" v-show="current_tab == 3">
        <label for="test-data-upload">Upload an image of a fruit salad...</label>
        <input type="file" ref="test_images" id="test-data-upload" accept="image/*"/>
        <button id="test-model" v-on:click="testModel">Load Image Data & Test Model</button>
        <div class="image-card">
          <img v-show="test_image" ref="uploaded_test_image" :src="test_image" alt="Have you uploaded a test image?">
          <p>{{ test_results }}</p>
        </div>
      </section>
      <section class="plain-container" v-show="current_tab == 4">
        welcome exportModel
      </section>
    </main>
    <br>
    <small>uses tf.js for ML</small>
    <div id="micro-out-div"></div>
  </div>
</template>

<script>
export default {
  name: 'ModelTraining',
  emits: ["done-exporting-model"],
  props: {
    header: String,
    objects: {
      required: true,
      type: Array,
      default: () => ([])
    },
    w: {
      required: true,
      type: Number,
    },
    h: {
      required: true,
      type: Number,
      default: () => ([])
    }
  },
  data() {
    return {
      tf: null,
      feature_extractor: null,
      fully_connected_head: null,
      training_dataset: [],
      test_image: undefined,
      test_results: "there are no results so far...",

      trainingProgress: 0,
      currentEpoch: 0,
      loss: null,
      acc: null,

      // FLAGS & COUNTERS
      current_tab: 1
    }
  },
  async mounted() {
    this.tf = window.tf;
    this.checkForTensorflow();

    await this.renderDatasetCards();
    await this.$nextTick();
    if(!this.fully_connected_head || !this.feature_extractor) await this.setupModels();
    await this.populateTrainingDataset();
  },
  methods: {
    checkForTensorflow() {
      if (this.tf) {
        console.log("tf.js is in this scope :D");
      } else {
        console.error("tf.js is NOT in this scope D':");
      }
    },
    async renderDatasetCards() {
      const res = await fetch("/assets/training-images/dataset-index.json");
      const image_filenames = await res.json();
      
      this.training_dataset = image_filenames.map((name) => {
        return {
          filename: name,
          tensor: undefined,
          labels: this.objects.map((label) => name.includes(label) ? 1 : 0)
        }
      });
    },
    async populateTrainingDataset() {
      console.log("training data before:-\n", this.training_dataset);

      console.log("File names:-\n");
      console.log(this.training_dataset.map((img) => `${img.filename}`).join(", \n"));

      const image_cards = this.$refs["image-card"];
      console.log("cards:", image_cards);
      for (let [i, el] of image_cards.entries()) {
        // console.log(i, this.$refs.dataset.children[i]);
        console.log(el.tagName);

        // assigning TENSOR
        const loadImage = (img) => new Promise((resolve, reject) => {
          if (img.complete && img.naturalWidth > 0) {
            resolve(img);
          } else {
            img.onload = () => resolve(img);
            img.onerror = (e) => reject(e);
          }
        });

        try {
          await loadImage(el.firstElementChild);
          this.training_dataset[i].tensor = this.preprocessImageData(el.firstElementChild);
        } catch (err) {
          console.error(`Error loading image at index ${i}:`, err);
        }
        this.training_dataset[i].tensor = this.preprocessImageData(el.firstElementChild);
      }

      console.log("training data after:-\n", this.training_dataset);
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

        return this.feature_extractor.predict(croppedImgTns).squeeze();
      });
    },
    async setupModels() {
      const NUM_CLASSES = this.objects.length;

      try {
        // loading the feature extractor model...
        const extractorURL =
          "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
        const loadedModel = await this.tf.loadLayersModel(extractorURL);

        // Creating the feature extractor base model...
        const beforeFinalLayer = loadedModel.getLayer(
          "global_average_pooling2d_1"
        );
        this.feature_extractor = this.tf.model({
          inputs: loadedModel.inputs,
          outputs: beforeFinalLayer.output,
        });

        // Warming up the feature extractor...
        this.tf.tidy(() => {
          const warmupInput = this.tf.zeros([1, this.w, this.h, 3]);
          const answer = this.feature_extractor.predict(warmupInput);
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
          inputShape: [this.feature_extractor.outputs[0].shape[1]],
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
      // const model_optimizer = this.tf.train.rmsprop(0.001);
      this.fully_connected_head.compile({
        optimizer: model_optimizer,
        loss: "binaryCrossentropy",
        metrics: [this.tf.metrics.binaryAccuracy],
      });

      console.log("Model created successfully");
      this.fully_connected_head.summary();
    },
    async fitClassifcationHead() {
      const xs = this.tf.stack(this.training_dataset.map((item) => item.tensor)); // tensor shape: [batch, height, width, channels]
      const ys = this.tf.stack(this.training_dataset.map((item) => item.labels)); // tensor shape: [batch, labels]

      console.log(xs.shape);
      console.log(ys.shape);

      this.tf.util.shuffleCombo(xs, ys);

      // const ls = this.training_dataset.map((item) => item.labels);
      // 
      // document.getElementById('micro-out-div').innerText = ls.map((item) => item.map((label, i) => label == 1 ? this.objects[i] : "_")).join("\n");

      const BATCH_SIZE = 8;
      const NUM_EPOCHS = 7;
      const start = performance.now();

      const results = await this.fully_connected_head.fit(xs, ys, {
        shuffle: true,
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            // update progress bar
            const percent = ((epoch + 1) / NUM_EPOCHS) * 100;
            this.trainingProgress = percent;

            // optional: show loss/accuracy live
            this.loss = logs.loss.toFixed(3);
            this.acc = logs.binaryAccuracy.toFixed(3);
          }
        }
      });

      const end = performance.now();
      const timeTakenInSeconds = ((end - start) / 1000).toFixed(2);
      console.log(`Training completed in ${timeTakenInSeconds} seconds`);

      // Clean up tensors
      xs.dispose();
      ys.dispose();

      console.log("Training results:", results.history);
    },
    async testModel() {
      const uploaded = Array.from(this.$refs.test_images.files);

      const img_element = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
          this.test_image = reader.result;

          this.$refs.uploaded_test_image.onload = () => resolve(this.$refs.uploaded_test_image);
          this.$refs.uploaded_test_image.onerror = reject;
        };

        reader.onerror = reject;
        reader.readAsDataURL(uploaded[0]);
      });

      const test_tensor = this.preprocessImageData(img_element);
      console.log("image tensor -", test_tensor);
      
      const predictions_list = await this.fully_connected_head.predict(test_tensor.expandDims(0)).array();
      console.log(predictions_list);
      console.log(this.objects);

      this.test_results = predictions_list[0].map((pred, i) => (pred >= 0.5) ? this.objects[i] : "").filter((s) => s !== "").join(", ") || "none";
      // this.test_results = predictions_list.map((_, i) => this.objects[i]);
    },
    async exportModel() {
      this.current_tab = 4;
      // save the model to browser's indexed database for inference
      await this.fully_connected_head.save("indexeddb://pretrained-head");

      this.$emit('done-exporting-model', true);
    }
  }
}
</script>

<style scoped>
/* 
old color palette : https://coolors.co/visualizer/331832-cc2936-1b5299-aaa694-cccccc

new color palette : https://coolors.co/visualizer/264653-2a9d8f-e9c46a-f4a261-e76f51
*/
.persian-green {
  background-color: #2A9D8F;

  color: black;
}
.saffron {
  background-color: #e9c46a;

  color: black;
}
.sandy-brown {
  background-color: #f4a261;

  color: black;
}

nav {
  display: flex;
  justify-content: center;
}

.tab {
  /* border-radius: 30% 30% 0% 0%; */
  border-radius: 15px 15px 0 0;
  padding: 10px 10px 5px;
  margin-right: 10px;

  /* width: 50px; */
  /* aspect-ratio: 1 / 1; */

  text-align: center;
  font-family: "Zain", sans-serif;
  font-size: 20px;
  font-weight: 800;

  /* background-clip: padding-box; */
}

.plain-container {
  border-width: 1px;
  border-style: solid;
  border-color: rgb(204, 204, 204);
  border-radius: 5px;
  margin: auto;
  padding-bottom: 20px;

  width: 90vw;

  background-color: white;
  box-shadow: 1px 1px 1px #999999;
}

.image-card {
  display: inline-block;

  border-width: 1px;
  border-style: solid;
  border-color: #999999;
  border-radius: 10px;
  margin: 20px 10px 0;
  padding: 10px;

  background-color: #cccccc66;
  box-shadow: 3px 3px 3px #666666;
}

.image-card img {
  width: 300px;
  height: 300px;

  margin: auto;
  
  border-width: 1px;
  border-style: solid;
  border-color: #999999;
}

.image-card p {
  border-radius: 999px;
  margin: auto;
  padding: 0 12px;

  width: fit-content;
  
  background-color: #cccccc;

  color: #000;
  text-align: center;
  font-family: "Zain", sans-serif;
  font-size: 16px;
  font-weight: 800;
}

button {
  padding: 8px;

  background-color: #264653;
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
