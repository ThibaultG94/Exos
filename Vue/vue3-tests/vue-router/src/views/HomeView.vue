<template>
	<div class="home">
		<!-- <img alt="Vue logo" src="../assets/logo.png">
    <HelloWorld msg="Welcome to Your Vue.js App"/> -->

		<div class="album py-5 bg-light">
			<div class="container">
				<div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3">
					<router-link
						v-for="menu in menus"
						:key="menu.id"
						:to="{
							name: 'menu.show',
							params: { id: menu.id, slug: menu.slug },
						}">
						<div class="col">
							<div class="card shadow-sm">
								<img
									:src="`/images/${menu.image}`"
									:alt="menu.name"
									width="338"
									height="275" />

								<div class="card-body">
									<h3>
										{{ menu.name }}
									</h3>
								</div>
							</div>
						</div>
					</router-link>
				</div>
			</div>
		</div>
	</div>
</template>

<script>
// @ is an alias to /src
// import HelloWorld from '@/components/HelloWorld.vue'

import menuData from '@/data.json';

export default {
	data() {
		return {
			menus: menuData.menus,
		};
	},
};
</script>
