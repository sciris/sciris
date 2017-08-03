<template>
  <div class="my-page">
    <h1>Scatterplotter for Vue</h1>

    <div id="app">
      <h2>Client request</h2>
      <label>Server info request</label>
      <input v-model="infoselect" />
      <button @click="sendRequest">Send request to server</button>
      <h2>Server return</h2>
      <p>{{ serverresponse }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
export default {
  name: 'MyPage',
  data () {
    return {
      infoselect: 'graph1',
      serverresponse: ''
    }
  },
  methods: {
    sendRequest () {
      // Use a GET request to pass along the value.
      /* var getResource = '/api?value=' + this.infoselect
      axios.get(getResource)
        .then(response => {
          this.serverresponse = response.data
        })
        .catch(error => {
          this.serverresponse = 'There was an error: ' + error.message
        }) */

      // Use a POST request to pass along the value.
      axios.post('/api', {
        value: this.infoselect
      })
      .then(response => {
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
