const application = {
	data() {
		return {
			title: 'Hello World',
			message: 'Voici ma première application Vue.js',
			side: 5,
			width: 500,
			height: 200,
		};
	},
	methods: {
		bonjour() {
			return 'Bonjour tout le monde';
		},
		calculSquareSurface() {
			return this.side * this.side;
		},
		calculRectangleSurface() {
			return this.width * this.height;
		},
	},
};

Vue.createApp(application).mount('#app');
