// index.js -- Vuex store configuration
//
// Last update: 9/13/17 (gchadder3)

import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const store = new Vuex.Store({
  state: {
    username: 'None'
  },

  mutations: {
    newuser (state, newname) {
      state.username = newname
    }
  }
})
export default store