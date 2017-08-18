<template>
  <div id="app">
    <h1>Scatterplotter for Vue</h1>
    <label>Server graph request</label>
    <input v-model="infoselect">
    <button @click="sendRequest">Fetch scatterplot</button>
    <p v-if='loadedfile'> 
      Following file loaded from server: {{ loadedfile }} 
    </p>
    <p v-if='servererror'> 
      Server could not find following file: {{ servererror }} 
    </p>
<!--    <p>{{ serverresponse }}</p> -->
    <div id="fig01"></div>
  </div>
</template>

<script>
import axios from 'axios'

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
      // Use a POST request to pass along the value of the graph to be found.
      axios.post('/api', {
        value: this.infoselect
      })
      .then(response => {
        // Pull out the response data.
        this.serverresponse = response.data

        // If we already have a figure, pop the figure object, and clear
        // the DOM.
        if (mpld3.figures.length > 0) {
          mpld3.figures.pop()
          document.getElementById('fig01').innerHTML = ''
        }

        // Clear the loaded file.
        this.loadedfile = ''

        // If there was no error returned...
        if (typeof(response.data.error) == 'undefined') {
          // Draw the figure in the 'fig01' div tag.
          mpld3.draw_figure('fig01', response.data)

          // Remember the file that was loaded.
          this.loadedfile = 'datafiles/' + this.infoselect + '.csv'

          // Clear the server error.
          this.servererror = ''

        // Otherwise (we got an error)...
        } else {
          // Clear the loaded file.
          this.loadedfile = ''

          // Set the server error.
          this.servererror = 'datafiles/' + this.infoselect + '.csv'
        }
      })
      .catch(error => {
        this.serverresponse = 'There was an error: ' + error.message
      })

      // Use a GET request to pass along the value.
      /* var getResource = '/api?value=' + this.infoselect
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

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
