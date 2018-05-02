<!-- 
MyPage.vue -- MyPage Vue component

Last update: 1/29/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <!-- Indicator if the user's directory is empty -->
    <p v-if='!selectedgraph'>
      Your directory is empty, so you need to upload a new file.
    </p>

    <!-- Selector for saved graph of user's -->
    <div v-if='selectedgraph'>
      <label>Scatterplot to retrieve</label>
      <select v-model='selectedgraph'>
        <option v-for='choice in resourcechoices'>{{ choice }}</option>
      </select>
      <button v-if='selectedgraph' @click="sendRequest">Load</button>
    </div>

    <!-- Button for uploading new files -->
    <button @click="uploadFile">Upload New File</button>
    <br/>

    <!-- Display of loaded in file along with buttons for downloading or 
    deleting -->
    <p v-if='loadedfile'> 
      Following file loaded from server: {{ loadedfile }} 
      <button @click="downloadFile">Download it!</button>
      <button @click="deleteFile">Delete it</button>
    </p>

    <!-- Display of any server error -->
    <p v-if='servererror'> 
      Server error: {{ servererror }} 
    </p>

<!--    <p>Server Response: {{ serverresponse }}</p> -->

    <!-- Window for mpld3 figure -->
    <div id="fig01"></div>
  </div>
</template>

<script>
import axios from 'axios'
var filesaver = require('file-saver')
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'MyPage',

  data () {
    return {
      selectedgraph: '',
      serverresponse: '',
      loadedfile: '', 
      servererror: '',
      resourcechoices: []
    }
  },

  created () {
    // If we have no user logged in, automatically redirect to the login page.
    if (this.$store.state.currentuser.displayname == undefined) {
      router.push('/login')
    } else {
      // Otherwise, get the list of the graphs available.   
      this.updateScatterplotDataList()
    }
  },

  methods: {
    clearFigureWindow () {
      // If we already have a figure, pop the figure object, and clear
      // the DOM.
      if (mpld3.figures.length > 0) {
        mpld3.figures.pop()
        document.getElementById('fig01').innerHTML = ''
      }
    },

    updateScatterplotDataList () {
      return new Promise((resolve, reject) => {
        rpcservice.rpcCall('list_saved_scatterplotdata_resources')
        .then(response => {
          this.resourcechoices = response.data
          this.selectedgraph = this.resourcechoices[0]
          resolve(response)
        })
        .catch(error => {
          reject(Error(response.data.error))
        })
      })
    },

    sendRequest () {
      // If we already have a figure, pop the figure object, and clear
      // the DOM.
      this.clearFigureWindow()

      // Clear the loaded file.
      this.loadedfile = ''

      // Clear the server error.
      this.servererror = ''

      // Call RPC get_saved_scatterplotdata_graph.
      rpcservice.rpcCall('get_saved_scatterplotdata_graph', [this.selectedgraph])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Draw the figure in the 'fig01' div tag.
        mpld3.draw_figure('fig01', response.data.graph)

        // Remember the file that was loaded.
        this.loadedfile = this.selectedgraph + '.csv'
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    }, 

    downloadFile () {
      // Clear the server error.
      this.servererror = ''

      // Call RPC download_saved_scatterplotdata.
      rpcservice.rpcDownloadCall('download_saved_scatterplotdata', [this.selectedgraph])
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
 
    deleteFile () {
      // Clear the server error.
      this.servererror = ''

      // Call RPC delete_saved_scatterplotdata_graph.
      rpcservice.rpcCall('delete_saved_scatterplotdata_graph', [this.selectedgraph])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Update the ScatterplotData list.
        this.updateScatterplotDataList().
        then(response => {
          // Clear the loaded file.
          this.loadedfile = ''

          // If we already have a figure, pop the figure object, and clear
          // the DOM.
          this.clearFigureWindow()
        })
      })
      .catch(error => {
        // Pull out the error message.
        this.serverresponse = 'There was an error: ' + error.message

        // Set the server error.
        this.servererror = error.message
      })
    },

    uploadFile () {
      // Clear the server error.
      this.servererror = ''

      // Call RPC upload_scatterplotdata_from_csv.
      rpcservice.rpcUploadCall('upload_scatterplotdata_from_csv', [this.selectedgraph], {}, '.csv')
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Update the ScatterplotData list.
        this.updateScatterplotDataList().
        then(response2 => {
          // Set the selected graph to the new upload.
          this.selectedgraph = response.data

          // Send a request to fetch the new file.
          this.sendRequest()
        })
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
