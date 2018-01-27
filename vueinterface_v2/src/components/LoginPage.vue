<!-- 
LoginPage.vue -- LoginPage Vue component

Last update: 1/22/18 (gchadder3)
-->

<template>
  <div class="LoginPage">
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
    <p>Login 3: Username = '_ScirisDemo', Password = '_ScirisDemo'</p>
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

          // Read in the full current user information.
          rpcservice.rpcGetCurrentUserInfo('get_current_user_info')
          .then(response2 => {
            // Set the username to what the server indicates.
            this.$store.commit('newuser', response2.data.user)

            // Navigate automatically to the home page.
            router.push('/')
          })
          .catch(error => {
            // Set the username to {}.  An error probably means the 
            // user is not logged in.
            this.$store.commit('newuser', {})
          })
        } else {
          // Set a failure result to show.
          this.loginResult = 'Login failed: username or password incorrect or account not activated.'
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
