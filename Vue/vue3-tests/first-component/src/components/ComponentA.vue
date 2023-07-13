<template>
	<div>
		<h1>
			{{ title }}
		</h1>
		<p>
			{{ message }}
		</p>
	</div>
</template>

<script>
export default {
	data() {
		return {
			title: 'Titre',
			message: 'Hello, je suis un message',
		};
	},
};
</script>

<style scoped></style>
