<template>
	<div class="container" id="app"></div>
	<ImageCard v-for="(card, index) in 3" :key="index" />
</template>

<script>
import ImageCard from './components/ImageCard.vue';

export default {
	name: 'App',
	components: {
		ImageCard,
	},
};
</script>

<style>
#app {
	font-family: Avenir, Helvetica, Arial, sans-serif;
	-webkit-font-smoothing: antialiased;
	-moz-osx-font-smoothing: grayscale;
	text-align: center;
	color: #2c3e50;
	margin-top: 60px;
}
</style>
