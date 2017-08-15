import Vue from 'vue'
import Router from 'vue-router'
import MyPage from '@/components/MyPage'
import Hello from '@/components/Hello'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'MyPage',
      component: MyPage
    },
    {
      path: '/vueinfo',
      name: 'Hello',
      component: Hello
    }
  ]
})
