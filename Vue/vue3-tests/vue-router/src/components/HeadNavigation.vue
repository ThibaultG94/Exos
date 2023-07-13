<template>
	<!-- <nav>
		<router-link to="/">Home</router-link> |
		<router-link to="/about">About</router-link> |
		<a href="/about">About 2</a>
	</nav> -->
	<nav class="navbar navbar-expand-md navbar-dark bg-dark mb-4">
		<div class="container-fluid">
			<router-link class="navbar-brand" to="/">Menu</router-link>
			<button
				class="navbar-toggler"
				type="button"
				data-bs-toggle="collapse"
				data-bs-target="#navbarCollapse"
				aria-controls="navbarCollapse"
				aria-expanded="false"
				aria-label="Toggle navigation">
				<span class="navbar-toggler-icon"></span>
			</button>
			<div class="collapse navbar-collapse" id="navbarCollapse">
				<ul class="navbar-nav me-auto mb-2 mb-md-0">
					<router-link
						v-for="menu in menus"
						:key="menu.id"
						:to="{
							name: 'menu.show',
							params: { id: menu.id, slug: menu.slug },
						}">
						<li class="nav-item">{{ menu.name }} | &nbsp;</li>
					</router-link>
					<!-- <li class="nav-item">
						<router-link class="nav-link" to="/">Home</router-link>
					</li>
					<li class="nav-item">
						<router-link class="nav-link" to="/menu-mc-burger"
							>Mc Do</router-link
						>
					</li>
					<li class="nav-item">
						<router-link class="nav-link" to="/menu-kfc-burger"
							>KFC</router-link
						>
					</li>
					<li class="nav-item">
						<router-link class="nav-link" to="/menu-bk-burger"
							>Burger King</router-link
						>
					</li> -->
				</ul>
			</div>
		</div>
	</nav>
</template>

<script>
import menuData from '@/data.json';

export default {
	data() {
		return {
			menus: menuData.menus,
		};
	},
};
</script>

<style scoped>
a.lien-actif {
	border-bottom: 2px solid #f55;
}
</style>
