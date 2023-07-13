let fruits = ['orange', 'pomme', 'citron', 'cerise', 'fraise'];

const application = {
	data() {
		return {
			fruits: fruits,
		};
	},
};

Vue.createApp(application).mount('#app');
