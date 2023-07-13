<template>
	<div>
		<!-- Menu view -->
		<!-- <h2>Menu {{ $route.params.id }}</h2> -->
		<h2>
			{{ menu.name }}
		</h2>
		<Back />
		<div>
			<img :src="`/images/${menu.image}`" :alt="menu.name" />
			<p>
				{{ menu.description }}
			</p>
		</div>
		<div class="ingredients">
			<router-link
				v-for="ingredient in menu.ingredients"
				:key="ingredient.slug"
				:to="{
					name: 'ingredient.show',
					params: { id: menu.id, slug: ingredient.slug },
				}">
				<IngredientsList :ingredient="ingredient" />
			</router-link>
		</div>
	</div>
</template>

<script>
import menuData from '@/data.json';
import IngredientsList from '@/components/IngredientsList.vue';
import Back from '@/components/Back.vue';

export default {
	props: {
		id: { type: Number, required: true },
	},
	computed: {
		// menuId() {
		// 	return parseInt(this.$route.params.id);
		// },
		menu() {
			return menuData.menus.find((menu) => menu.id === this.id);
		},
	},
	components: { IngredientsList, Back },
};
</script>

<style>
.ingredients {
	display: inline-flex;
}
</style>
