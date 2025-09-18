<template>
  <div class="training">
    <h2>{{ header }}</h2>
    <button id="inference-button" v-on:click="run">Run Predictor</button>
    <br>
    <br>
    <div class="image-container">
        <div id="example1" :title="imageTooltips[0]" class="example-pics"></div>
        <div id="example2" :title="imageTooltips[1]" class="example-pics"></div>
        <div id="example3" :title="imageTooltips[2]" class="example-pics"></div>
        <div id="example4" :title="imageTooltips[3]" class="example-pics"></div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Predictor',
  props: {
    header: String,
    w: {type:Number, required: true},
    h: {type:Number, required: true}
  },
  mounted() {
    this.tf = window.tf;
  },
  data() {
    return {
      imageTooltips: [
        "No predictions made yet", 
        "No predictions made yet", 
        "No predictions made yet", 
        "No predictions made yet"
      ]
    }
  },
  methods: {
    // Checks if tf.js is available in the component, as the name suggests
    checkForTensorflow() {
      if (this.tf) {
        console.log("tf.js IS in this scope :D");
      } else {
        console.error("tf.js is NOT in this scope D':");
      }
    },
    // given a div element's id, it extracts a tensor from the div's background image
    getInputTensor(id, feature_extractor) {
      const divElement = document.getElementById(id);
      const divStyles = getComputedStyle(divElement);
      const imageUrl = divStyles.backgroundImage;
      const intWidth = parseInt(divStyles.width.replace("px", ""));
      const intHeight = parseInt(divStyles.height.replace("px", ""));

      const img = new Image();
      img.src = imageUrl.slice(5, -2); // .slice() removes the url("") wrapper text;

      const canvas = document.createElement('canvas');
      canvas.width = intWidth;

      canvas.height = intHeight;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, intWidth, intHeight);

      const raw_image_tensor = this.tf.browser.fromPixels(canvas);
        
      // setting crop size operation...
      const widthToHeight = raw_image_tensor.shape[1] / raw_image_tensor.shape[0];
      let croppedSize;

      if (widthToHeight > 1) {
        // Image is wider than tall? crop sides
        const heightToWidth = raw_image_tensor.shape[0] / raw_image_tensor.shape[1];
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
      const suitable_tensor = this.tf.image.cropAndResize(
        raw_image_tensor.expandDims(0),
        croppedSize,
        [0],
        [this.w, this.h]
      )
      .toFloat()
      .div(255.0);
      
      // disposal/s
      raw_image_tensor.dispose();
      
      return feature_extractor.predict(suitable_tensor).squeeze();
    },
    async loadModel() {
      // const model = await this.tf.loadLayersModel('/assets/pretrained head/pretrained-head.json'); // new ERROR with loading caused by movement of files
      const model = await this.tf.loadLayersModel('indexeddb://pretrained-head');

      console.log("Model loaded successfully, YAY!")
      model.summary();

      return model;
    },
    makeInferences(trainedModel, imgTensors) {
      const output = trainedModel.predictOnBatch(imgTensors);
      return output;
    },
    async decodePredictions(preds, threshold=0.5, legend = {"strawberry":0, "orange":1, "kiwi":2}) {
      // converting model predictions to array of probabilities...
      const probabilities = await preds.array();

      this.imageTooltips = probabilities.map((inference) => {
        let tooltip_array = [];
        for(let [label, index] of Object.entries(legend)) {
          if(inference[index] >= threshold) tooltip_array.push("detected "+label);
        }
        return tooltip_array.join(" & ");
      });
      
      return this.imageTooltips;
    },
    async run() {
      // loading feature extractor
      const MobileNetV2URL =
        "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";

      // Load the MobileNet model using Google's optimized URL (like the fast example)
      const directlyLoaded = await this.tf.loadLayersModel(MobileNetV2URL);

      // Create feature extraction model (like the fast example)
      const beforeFinalLayer = directlyLoaded.getLayer(
        "global_average_pooling2d_1"
      );

      const loadedModel = this.tf.model({
        inputs: directlyLoaded.inputs,
        outputs: beforeFinalLayer.output,
      });

      console.log("MobileNet feature extractor loaded");

      // Warm up the model with zeroes (for memory optimization)
      this.tf.tidy(() => {
        const warmupInput = this.tf.zeros([1, 224, 224, 3]);
        const answer = loadedModel.predict(warmupInput);
        console.log("Warmup completed, shape of extracted features:", answer.shape);
      });
      
      const tensors_batch = [];
      for(let ex of ['example1', 'example2', 'example3', 'example4']) {
        tensors_batch.push(this.getInputTensor(ex, loadedModel));
      }
      const batch_tensor = this.tf.stack(tensors_batch);

      const predictor = await this.loadModel();
      const inferences = this.makeInferences(predictor, batch_tensor);

      inferences.print();
      console.log("decoded:", await this.decodePredictions(inferences));
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
a {
  color: #42b983;
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

div.image-container {
  display: flex;
  flex-flow: row wrap;
  justify-content: center;
  align-content: stretch;

  width: 97vw;
  padding: 1vw;
  /* background-color: #08415C; */
  background-size: 100% 100%;
}

div.example-pics {
  width: 224px;
  height: 224px;

  margin-left: 10px;
  margin-top: 10px;

  background-repeat: no-repeat;
  background-size: 100% 100%;
  border-width: 5px;
  border-style: groove;
  border-color: #e9e9e9;

}
#example1 {
  background-image: url('/assets/test-images/fruit-salad-01.jpg');
}
#example2 {
  background-image: url('/assets/test-images/fruit-salad-02.jpg');
}
#example3 {
  background-image: url('/assets/test-images/fruit-salad-03.jpg');
}
#example4 {
  background-image: url('/assets/test-images/sliced-oranges.webp');
}
</style>
