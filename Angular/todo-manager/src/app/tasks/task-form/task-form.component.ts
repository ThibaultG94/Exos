import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TaskService } from '../../services/task.service';

@Component({
  selector: 'app-task-form',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './task-form.component.html',
  styleUrl: './task-form.component.scss'
})
export class TaskFormComponent {
  newTask = {
    title: '',
    description: '',
    dueDate: ''
  };

  constructor(private taskService: TaskService) {}

  onSubmit(): void {
    if (this.newTask.title.trim()) {
      this.taskService.addTask({
        title: this.newTask.title,
        description: this.newTask.description,
        dueDate: this.newTask.dueDate ? new Date(this.newTask.dueDate) : undefined,
        completed: false
      });

      // Réinitialiser le formulaire
      this.newTask = {
        title: '',
        description: '',
        dueDate: ''
      };
    }
  }
}