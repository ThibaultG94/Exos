import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderComponent } from './shared/header/header.component';
import { TaskListComponent } from './tasks/task-list/task-list.component';
import { TaskFormComponent } from './tasks/task-form/task-form.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    HeaderComponent,
    TaskListComponent,
    TaskFormComponent
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'todo-manager';
}