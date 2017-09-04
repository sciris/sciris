<!-- 
MyPage.vue -- MyPage Vue component

Last update: 9/4/17 (gchadder3)
-->

<template>
  <div id="app">
    <h1>Scatterplotter for Vue</h1>

    <label>Server graph request</label>
    <select v-model='infoselect'>
      <option v-for='choice in resourcechoices'>{{ choice }}</option>
    </select>
<!--    <input v-model='infoselect'/> -->
    <button @click="sendRequest">Fetch scatterplot</button>
    <br/>

    <button @click="uploadFile">File Upload</button>
    <br/>

    <p v-if='loadedfile'> 
      Following file loaded from server: {{ loadedfile }} 
      <button @click="downloadFile">Download it!</button>
    </p>

    <p v-if='servererror'> 
      Server error: {{ servererror }} 
    </p>

<!--    <p>Server Response: {{ serverresponse }}</p> -->

    <div id="fig01"></div>
  </div>
</template>

<script>
import axios from 'axios'
var filesaver = require('file-saver')
import rpcservice from '../services/rpc-service'

export default {
  name: 'MyPage',
  data () {
    return {
      infoselect: 'graph1',
      serverresponse: '',
      loadedfile: '', 
      servererror: '',
      resourcechoices: ['graph1', 'graph2', 'graph3', 'graph4']
    }
  },

  created () {
    this.updateScatterplotDataList()
  },

  methods: {
    updateScatterplotDataList () {
      rpcservice.rpcCall('list_saved_scatterplotdata_resources')
      .then(response => {
        this.resourcechoices = response.data
      })
    },

    sendRequest () {
      // If we already have a figure, pop the figure object, and clear
      // the DOM.
      if (mpld3.figures.length > 0) {
        mpld3.figures.pop()
        document.getElementById('fig01').innerHTML = ''
      }

      // Clear the loaded file.
      this.loadedfile = ''

      // Clear the server error.
      this.servererror = ''

      // Call RPC get_saved_scatterplotdata_graph.
      rpcservice.rpcCall('get_saved_scatterplotdata_graph', [this.infoselect])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Draw the figure in the 'fig01' div tag.
        mpld3.draw_figure('fig01', response.data)

        // Remember the file that was loaded.
        this.loadedfile = 'datafiles/' + this.infoselect + '.csv'
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    }, 

    downloadFile () {
      // Call RPC download_saved_scatterplotdata.
      rpcservice.rpcDownloadCall('download_saved_scatterplotdata', [this.infoselect])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    }, 

    uploadFile () {
      // Call RPC upload_scatterplotdata_from_csv.
      rpcservice.rpcUploadCall('upload_scatterplotdata_from_csv', [this.infoselect], {}, '.csv')
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Update the ScatterplotData list.
        this.updateScatterplotDataList()
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message 
      })  
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
