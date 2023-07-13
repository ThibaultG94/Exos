<template>
	<div class="hello">
		<p
			:class="text1Clicked ? 'red' : null"
			v-text="text1"
			@click="text1Clicked = !text1Clicked"></p>
		<p>
			{{ getClassText1() }}
		</p>
		<p
			:class="text2Clicked ? 'blue' : null"
			v-text="text2"
			@click="text2Clicked = !text2Clicked"></p>
		<p>
			{{ getClassText2 }}
		</p>
	</div>
</template>

<script>
export default {
	name: 'HelloWorld',
	data() {
		return {
			text1: 'Hello',
			text2: 'Bye Bye',
			text1Clicked: false,
			text2Clicked: false,
		};
	},
	methods: {
		getClassText1() {
			console.log('Text1 modifié');
			return this.text1Clicked
				? 'Class de text1: red'
				: 'Text1 sans classe';
		},
	},
	computed: {
		getClassText2() {
			console.log('Text2 modifié');
			return this.text2Clicked
				? 'Class de text2: blue'
				: 'Text2 sans classe';
		},
	},
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.red {
	color: red;
}
.blue {
	color: blue;
}
h3 {
	margin: 40px 0 0;
}
ul {
	list-style-type: none;
	padding: 0;
}
li {
	display: inline-block;
	margin: 0 10px;
}
a {
	color: #42b983;
}
</style>
