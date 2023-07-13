const application = {
	data() {
		return {
			textHTML: '<p class="text-danger">Attention !</p>',
		};
	},
};

Vue.createApp(application).mount('#app');
