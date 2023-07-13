<template>
	<div v-if="this.responseData" class="row d-inline-flex">
		<div
			v-for="(data, index) in responseData"
			:key="index"
			class="card"
			style="width: 18rem">
			<img :src="data.thumbnailUrl" class="card-img-top" alt="" />
			<div class="card-body">
				<h5 class="card-title text-truncate">
					{{ data.title }}
				</h5>
				<a :href="data.url" class="btn btn-primary">
					Voir l'image {{ data.id }}
				</a>
			</div>
		</div>
	</div>
</template>

<script>
export default {
	name: 'PhotosRow',

	data() {
		return {
			apiUrl: 'https://jsonplaceholder.typicode.com/photos',
			responseData: null,
		};
	},

	methods: {
		fetchApi() {
			this.axios
				.get(this.apiUrl)
				.then((res) => {
					this.responseData = res.data;
					console.log(res.data);
				})
				.catch((error) => {
					console.log(error);
				});
		},
	},

	mounted() {
		this.fetchApi();
	},
};
</script>

<style scoped>
.card {
	margin: 15px;
}
</style>
