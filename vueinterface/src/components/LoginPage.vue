<!-- 
LoginPage.vue -- LoginPage Vue component

Last update: 9/14/17 (gchadder3)
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

    <p>Login 1: Username = 'newguy', Password = 'mesogreen'</p>
    <p>Login 2: Username = 'admin', Password = 'mesoawesome'</p>
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'
import router from '../router'

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
      .then(response => {
        if (response.data == 'success') {
          // Set a success result to show.
          this.loginResult = 'Success!'

          // Update the username.
          this.$store.commit('newuser', this.loginUserName)

          // Navigate automatically to the home page.
          router.push('/')
        } else {
          // Set a failure result to show.
          this.loginResult = 'Login failed: username or password incorrect.'
        }
      })
      .catch(error => {
        this.loginResult = 'Server error.  Please try again later.'
      })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
