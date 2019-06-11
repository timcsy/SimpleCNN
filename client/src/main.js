import Vue from 'vue';
import App from './app.vue'; 
new Vue({
    el: '#app',
    components: {
        App
    },
    
    render: (createElement) => createElement(App)
});