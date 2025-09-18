<template>
  <div id="app">
    <h1>Multi-label Image Classfication Demo</h1>
    <Predictor 
      v-if="use_pretrained_classifier_head" 
      header="Model Inference" 
      :w="inputSize.w" 
      :h="inputSize.h"
    />
    <Training 
      v-else 
      ref="trainingComponent"
      header="Model Training" 
      :objects="['strawberry', 'orange', 'kiwi']"
      :w="inputSize.w" 
      :h="inputSize.h"
      @done-exporting-model="use_pretrained_classifier_head = $event"
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
      use_pretrained_classifier_head: false
    }
  },
  components: {
    Training: ModelTraining,
    Predictor
  }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Zain:ital,wght@0,200;0,300;0,400;0,700;0,800;0,900;1,300;1,400&display=swap');

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
