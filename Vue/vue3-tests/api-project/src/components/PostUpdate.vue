<template>
	<div class="row d-linline-flex">
		<form @submit.prevent="onSubmit">
			<div class="form-group">
				<input
					type="text"
					name="title"
					v-model="this.title"
					ref="put_title"
					class="form-control"
					placeholder="Titre" />
			</div>
			<br />
			<div class="form-group">
				<textarea
					name="body"
					v-model="this.body"
					ref="put_body"
					class="form-control"
					placeholder="Body..."></textarea>
			</div>
			<br /><br />
			<input
				type="hidden"
				name="userId"
				v-model="this.userId"
				ref="put_userId" />
			<input type="hidden" name="id" v-model="this.id" ref="put_id" />
			<div class="form-group">
				<button class="btn btn-primary">Update</button>
			</div>
		</form>
	</div>
</template>

<script>
// import axios from 'axios';

export default {
	name: 'PostUpdate',

	data() {
		return {
			apiUrl: 'https://jsonplaceholder.typicode.com/posts/1',
			id: '',
			title: '',
			body: '',
			userId: '',
		};
	},
	methods: {
		onSubmit() {
			console.log(this.$refs);
			const { put_id, put_title, put_body, put_userId } = this.$refs;
			const id = put_id.value;

			let body = JSON.stringify({
				id: id,
				title: put_title.value,
				body: put_body.value,
				userId: put_userId.value,
			});

			this.axios
				.put('https://jsonplaceholder.typicode.com/posts/' + id, body)
				.then((response) => {
					console.log(response);
					console.log(response.status);
					if ((response.status = 200)) {
						console.log(
							'Post mis à jour : ' + JSON.stringify(response.data)
						);
					}
				});
		},
	},
	mounted() {
		this.axios.get(this.apiUrl).then((response) => {
			console.log(response);
			this.title = response.data.title;
			this.body = response.data.body;
			this.id = response.data.id;
			this.userId = response.data.userId;
		});
	},
};
</script>

<style scoped></style>
