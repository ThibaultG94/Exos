<template>
	<img alt="Vue logo" src="./assets/logo.png" />
	<SuperButton>Submit</SuperButton>
	<br />
	<super-button></super-button>
	<br />
	<SuperButton>
		<span style="background-color: #f55">Cliquez ici</span>
		<IconButton></IconButton>
	</SuperButton>
	<br />
	<HeaderMenu>
		<!-- <template v-slot:alert> -->
		<template #alert>
			<div class="alert alert-success">Message de succès</div>
		</template>
	</HeaderMenu>
</template>

<script>
import SuperButton from './components/SuperButton.vue';
import IconButton from './components/IconButton.vue';
import HeaderMenu from './components/HeaderMenu.vue';

export default {
	name: 'App',
	components: { SuperButton, IconButton, HeaderMenu },
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
