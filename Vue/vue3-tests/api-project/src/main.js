import { createApp } from 'vue';
import App from './App.vue';
import axios from 'axios';
import VueAxios from 'vue-axios';

const VueApp = createApp(App);
VueApp.use(VueAxios, axios).mount('#app');

import 'bootstrap/dist/css/bootstrap.css';
