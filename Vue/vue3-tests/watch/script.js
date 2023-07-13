const application = {
	data() {
		return {
			km: 0,
			finish: 5,
		};
	},
	methods: {
		drive() {
			setInterval(() => {
				this.km++;
			}, 1000);
		},
	},
	watch: {
		km() {
			if (this.km === this.finish) {
				this.km = 0;
				alert('Vous êtes arrivé à destination');
			}
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
