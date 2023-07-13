<template>
	<HeadNavigation />

	<div class="container">
		<router-view v-slot="{ Component }">
			<!-- <transition name="slide-fade"> -->
			<!-- <transition name="bounce"> -->
			<transition :name="$route.meta.transition">
				<component :is="Component" :key="$route.path"></component>
			</transition>
		</router-view>
	</div>
</template>

<script>
import HeadNavigation from '@/components/HeadNavigation.vue';

export default {
	components: { HeadNavigation },
};
</script>

<style>
#app {
	font-family: Avenir, Helvetica, Arial, sans-serif;
	-webkit-font-smoothing: antialiased;
	-moz-osx-font-smoothing: grayscale;
	text-align: center;
	color: #2c3e50;
}

.bounce-enter-active {
	animation: bounce-in 0.5s;
}

.bounce-leave-active {
	animation: bounce-in 0.5s reverse;
}

@keyframes bounce-in {
	0% {
		transform: scale(0);
	}
	50% {
		transform: scale(1.25);
	}
	100% {
		transform: scale(1);
	}
}

/* .slide-fade-enter-active {
	transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
	transition: all 0.8s cubic-bezier(1, 0.5, 0.8, 1);
}

.slide-fade-enter-from,
.slide-fade-leave-to {
	transform: translateX(20px);
	opacity: 0;
} */

nav {
	padding: 30px;
}

nav a {
	font-weight: bold;
	color: #2c3e50;
}

nav a.router-link-exact-active {
	color: #42b983;
}
</style>
