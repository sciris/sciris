<template>
  <div id="app">
    <h1>Random Scatterplotter for Vue</h1>
    <button @click="sendRequest">Get random scatterplot</button>
    <div id="fig01"></div>
  </div>

  <!-- old version
  <div id="app">
    <h1>Scatterplotter for Vue</h1>
    <h2>Client request</h2>
    <label>Server info request</label>
    <input v-model="infoselect">
    <button @click="sendRequest">Send request to server</button>
    <h2>Server return</h2>
    <p>{{ serverresponse }}</p>
    <div id="fig01"></div>
  </div>
  -->
</template>

<script>
import axios from 'axios'
// require('script-loader!./d3.v3.min.js')
// require('script-loader!./mpld3.v0.3.js')
// require('script-loader!./testlog.js')
// import exec from './d3.v3.min.js'
// import exec2 from './mpld3.v0.3.js'
// import exec3 from './testlog.js'
export default {
  name: 'app',
  data () {
    return {
      infoselect: 'graph1',
      serverresponse: ''
    }
  },
  methods: {
    sendRequest () {
      // testing code for importing legacy JS libraries
      // testlog()

      // Use a POST request to pass along the value.
      axios.post('/api', {
        value: this.infoselect
      })
      .then(response => {
        this.serverresponse = response.data
        // If we already have a figure, pop the figure object, and clear 
        // the DOM.
        if (mpld3.figures.length > 0) {
          mpld3.figures.pop()
          document.getElementById('fig01').innerHTML = ''
        }
        mpld3.draw_figure('fig01', response.data)
      })
      .catch(error => {
        this.serverresponse = 'There was an error: ' + error.message
      })

      // Use a GET request to pass along the value.
      /* var getResource = 'http://localhost:8080/api?value=' + this.infoselect
      axios.get(getResource)
        .then(response => {
          this.serverresponse = response.data
        })
        .catch(error => {
          this.serverresponse = 'There was an error: ' + error.message
        }) */
    }
  }
}
</script>

<style>
</style>
