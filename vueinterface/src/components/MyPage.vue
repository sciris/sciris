<template>
  <div id="app">
    <h1>Scatterplotter for Vue</h1>

    <label>Server graph request</label>
    <input v-model="infoselect"/>
    <button @click="sendRequest">Fetch scatterplot</button>
    <br/>

    <form enctype="multipart/form-data">
      <label>File to upload to server:</label>
      <input type="file" name="uploadfile" accept=".csv" @change="onFileChange"/>
    </form>

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
      servererror: ''
    }
  },
  methods: {
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

      // Call RPC get_graph_from_file.
      rpcservice.rpcCall('get_graph_from_file', [this.infoselect])
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // Draw the figure in the 'fig01' div tag.
        mpld3.draw_figure('fig01', response.data)

        // Remember the file that was loaded.
        this.loadedfile = 'datafiles/' + this.infoselect + '.csv'
      })
      .catch(error => {
        this.serverresponse = 'There was an error: ' + error.message

        // Clear the server error.
        this.servererror = error.message
      })
    }, 

    downloadFile () {
      // Use a POST request to pass along the value of the graph to be found.
      axios.post('/api/download', 
        {
          value: this.infoselect
        },
        {
          responseType: 'blob'
        })
      .then(response => {
        // Create a new blob object (containing the file data) from the
        // response.data component.
        var blob = new Blob([response.data])

        // Grab the file name from response.headers.
        var filename = response.headers.filename

        // Bring up the browser dialog allowing the user to save the file 
        // or cancel doing so.
        filesaver.saveAs(blob, filename)

        // Pull out the response data.
        this.serverresponse = response
        // this.serverresponse = response.headers
      })
      .catch(error => {
        this.serverresponse = 'There was an error: ' + error.message
      })
    },

    onFileChange (e) {
      var files = e.target.files || e.dataTransfer.files
      if (!files.length)
        return

      const formData = new FormData()
      formData.append(e.target.name, files[0])

      // Use a POST request to pass along file to the server.
      axios.post('/api/upload', formData)
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data
      })
      .catch(error => {
        this.serverresponse = 'There was an error: ' + error.message
      })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
