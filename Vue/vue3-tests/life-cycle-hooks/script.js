const application = {
	data() {
		return {
			km: 0,
		};
	},
	methods: {
		drive() {
			setInterval(() => {
				this.km++;
			}, 1000);
		},
	},
	beforeCreate() {
		console.log(this.km);
	},
	created() {
		this.drive();
	},
};

Vue.createApp(application).mount('#app');
