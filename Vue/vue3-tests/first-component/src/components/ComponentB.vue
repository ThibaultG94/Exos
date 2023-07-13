<template>
	<div>
		<p>
			{{ message }}
		</p>
		<button type="button" @click="increment()">
			{{ counter }}
		</button>
	</div>
</template>

<script>
export default {
	data() {
		return {
			message: 'Je suis un message du composant B',
			counter: 0,
		};
	},
	methods: {
		increment() {
			this.counter++;
		},
	},
};
</script>

<style></style>
