import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';

// Importe les composants
import { HeaderComponent } from './shared/header/header.component';
import { TaskListComponent } from './tasks/task-list/task-list.component';
import { TaskFormComponent } from './tasks/task-form/task-form.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    TaskListComponent,
    TaskFormComponent
  ],
  imports: [
    BrowserModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }