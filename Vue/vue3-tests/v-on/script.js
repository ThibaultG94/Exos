const application = {
	data() {
		return {};
	},
	methods: {
		redButton() {
			return alert("Tu as cliqué sur le bouton rouge, c'est interdit :(");
		},
		blueButton() {
			return alert("Tu as cliqué sur le bouton bleu, c'est cool :)");
		},
		blackButton() {
			return alert('La souris est sortie du bouton noir :o');
		},
	},
};

Vue.createApp(application).mount('#app');
