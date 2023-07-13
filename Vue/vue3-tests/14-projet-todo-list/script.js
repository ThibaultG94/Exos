// var tasks = [
// 	{
// 		name: 'Faire les courses',
// 		done: false,
// 	},
// 	{
// 		name: 'Coder mon projet',
// 		done: true,
// 	},
// ];

const todoList = {
	data() {
		return {
			tasks: [],
			newTask: {
				done: false,
			},
			errors: [],
		};
	},
	methods: {
		addTask() {
			this.errors = [];

			if (!this.newTask.name) {
				this.errors.push('Veuillez ajouter une tâche');
				return false;
			}

			this.tasks.push(this.newTask);
			this.newTask = {};

			this.$refs.name.focus();

			// localStorage.setItem('tasks', JSON.stringify(this.tasks));
		},
		removeAll() {
			if (!confirm('Confirmez vous la suppression ?')) {
				return false;
			}
			this.tasks = [];
		},
		deleteTask(task) {
			if (!confirm('Confirmez vous la suppression ?')) {
				return false;
			}

			const newTaskList = this.tasks.filter((el) => el !== task);
			this.tasks = newTaskList;
		},
	},
	created() {
		this.tasks = localStorage.getItem('tasks')
			? JSON.parse(localStorage.getItem('tasks'))
			: this.tasks;
	},
	mounted() {
		this.$refs.name.focus();
	},
	updated() {
		localStorage.setItem('tasks', JSON.stringify(this.tasks));
	},
};

Vue.createApp(todoList).mount('#app');
