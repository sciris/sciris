<!-- 
RegisterPage.vue -- RegisterPage Vue component

Last update: 2/13/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <label>Username:</label>
    <input v-model='registerUserName'/>
    <br/>

    <label>Password:</label>
    <input v-model='registerPassword'/>
    <br/>

    <label>Display Name:</label>
    <input v-model='registerDisplayName'/>
    <br/>

    <label>Email:</label>
    <input v-model='registerEmail'/>
    <br/>

    <button @click="tryRegister">Register</button>
    <br/>

    <p v-if="registerResult != ''">{{ registerResult }}</p>

    Already have an account?
    <router-link to="/login">
        Log in
    </router-link> 
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'RegisterPage', 

  data () {
    return {
      registerUserName: '',
      registerPassword: '',
      registerDisplayName: '',
      registerEmail: '',
      registerResult: ''
    }
  }, 

  methods: {
    tryRegister () {
      rpcservice.rpcRegisterCall('user_register', this.registerUserName, 
        this.registerPassword, this.registerDisplayName, this.registerEmail)
      .then(response => {
        if (response.data == 'success') {
          // Set a success result to show.
          this.registerResult = 'Success!'

          // Navigate automatically to the login page.
          router.push('/login')
        } else {
          // Set a failure result to show.
          this.registerResult = 'Registration failed.  Try again, possibly with a different username.'
        }
      })
      .catch(error => {
        this.registerResult = 'Server error.  Please try again later.'
      })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
