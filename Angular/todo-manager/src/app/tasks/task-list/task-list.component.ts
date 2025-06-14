import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Task } from '../../models/task.model';
import { TaskService } from '../../services/task.service';

@Component({
  selector: 'app-task-list',
  standalone: true,
  imports: [CommonModule],  // Ceci inclut le DatePipe
  templateUrl: './task-list.component.html',
  styleUrl: './task-list.component.scss'
})
export class TaskListComponent implements OnInit {
  tasks: Task[] = [];

  constructor(private taskService: TaskService) {}

  ngOnInit(): void {
    this.taskService.getTasks().subscribe(tasks => {
      this.tasks = tasks;
    });
  }

  toggleComplete(task: Task): void {
    task.completed = !task.completed;
    this.taskService.updateTask(task);
  }

  deleteTask(id: number): void {
    this.taskService.deleteTask(id);
  }
}