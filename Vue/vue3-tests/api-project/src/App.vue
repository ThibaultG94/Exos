<template>
	<div id="app" class="container">
		<!-- <PhotosRow /> -->
		<!-- <PostStore /> -->
		<!-- <PostUpdate /> -->
		<PostDelete />
	</div>
</template>

<script>
// import PhotosRow from './components/PhotosRow.vue';
// import PostStore from './components/PostStore.vue';
// import PostUpdate from './components/PostUpdate.vue';
import PostDelete from './components/PostDelete.vue';

export default {
	name: 'App',
	components: {
		// PhotosRow
		// PostStore,
		// PostUpdate,
		PostDelete,
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
