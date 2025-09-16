<template>
  <div id="app">

    <input type="checkbox" id="predictorCheckbox" v-model="doneTraining">
    <label for="predictorCheckbox">Already saved a pre-trained model</label>
    <h1>Multi-label Image Classfication Demo</h1>
    <Predictor 
      v-if="doneTraining" 
      header="Model Inference" 
      :w="inputSize.w" 
      :h="inputSize.h"
    />
    <Training 
      v-else 
      ref="trainingComponent"
      header="Model Training" 
      :w="inputSize.w" 
      :h="inputSize.h"
      @training-is-complete="doneTraining = $event"
    />

  </div>
</template>

<script>
import ModelTraining from './components/ModelTrainingComponent.vue'
import Predictor from './components/PredictorComponent.vue'

export default {
  name: 'App',
  data() {
    return {
      inputSize: {w:224, h:224}, // previously {w:512, h:512}

      // FLAGS & COUNTERS
      doneTraining: false
    }
  },
  components: {
    Training: ModelTraining,
    Predictor
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
