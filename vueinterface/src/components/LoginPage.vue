<!-- 
LoginPage.vue -- LoginPage Vue component

Last update: 9/9/17 (gchadder3)
-->

<template>
  <div id="app">
    <label>Username:</label>
    <input v-model='loginUserName'/>
    <br/>

    <label>Password:</label>
    <input v-model='loginPassword'/>
    <br/>

    <button @click="tryLogin">Login</button>
    <br/>

    <p v-if="loginResult != ''">{{ loginResult }}</p>

    <p>Hint: Username = 'newguy', Password = 'mesogreen'</p>
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'

export default {
  name: 'LoginPage', 

  data () {
    return {
      loginUserName: '',
      loginPassword: '',
      loginResult: ''
    }
  }, 

  methods: {
    tryLogin () {
      rpcservice.rpcLoginCall('user_login', this.loginUserName, this.loginPassword)
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
