import { Injectable } from '@angular/core';
import { Task } from '../models/task.model';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TaskService {
  private tasks: Task[] = [];
  private tasksSubject = new BehaviorSubject<Task[]>([]);

  getTasks(): Observable<Task[]> {
    return this.tasksSubject.asObservable();
  }

  addTask(task: Omit<Task, 'id'>): void {
    const newTask = {
      ...task,
      id: this.generateId()
    };
    this.tasks.push(newTask);
    this.tasksSubject.next([...this.tasks]);
  }

  updateTask(updatedTask: Task): void {
    const index = this.tasks.findIndex(task => task.id === updatedTask.id);
    if (index !== -1) {
      this.tasks[index] = updatedTask;
      this.tasksSubject.next([...this.tasks]);
    }
  }

  deleteTask(id: number): void {
    this.tasks = this.tasks.filter(task => task.id !== id);
    this.tasksSubject.next([...this.tasks]);
  }

  private generateId(): number {
    return this.tasks.length > 0 
      ? Math.max(...this.tasks.map(task => task.id)) + 1 
      : 1;
  }
}
