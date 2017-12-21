// index.js -- Vuex store configuration
//
// Last update: 9/23/17 (gchadder3)

import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const store = new Vuex.Store({
  state: {
    currentuser: {}
  },

  mutations: {
    newuser (state, user) {
      state.currentuser = user
    }
  }
})
export default store