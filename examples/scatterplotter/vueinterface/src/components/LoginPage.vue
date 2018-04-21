<!-- 
LoginPage.vue -- LoginPage Vue component

Last update: 2/14/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <form name="LogInForm" @submit.prevent="tryLogin" 
          style="max-width: 500px; min-width: 200px; margin: 0 auto">

      <h1>Log in</h1>

      <div class="modal-body">
        <div class="section" v-if="loginResult != ''">{{ loginResult }}</div>

        <div class="section form-input-validate">
          <input class="txbox __l"
                 type="text"
                 name="username"
                 placeholder="User name"
                 required="required"
                 v-model='loginUserName'/>
        </div>

        <div class="section form-input-validate">
          <input class="txbox __l"
                 type="password"
                 name="password"
                 placeholder="Password"
                 required="required"
                 v-model='loginPassword'/>
        </div>

        <button type="submit" class="section btn __l __block">Login</button>

        <div class="section">
          New user?
          <router-link class="link __blue" to="/register">
            Register here
          </router-link> 
        </div>

        <p>Login 1: Username = 'newguy', Password = 'mesogreen'</p>
        <p>Login 2: Username = 'admin', Password = 'mesoawesome'</p>
        <p>Login 3: Username = '_ScirisDemo', Password = '_ScirisDemo'</p>

      </div>
    </form>
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
          this.loginResult = 'Logging in...'

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
