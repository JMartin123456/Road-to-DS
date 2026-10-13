package com.example.demo;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;


@RestController
public class TaskController {

	private List<Task> todoList = new ArrayList<>();
	
	@GetMapping("/tasks")
	public List<Task> showTasks() {
		return todoList;
	}

	@GetMapping("/tasks/{id}")
	public Task showTask(@PathVariable int id) {
		return todoList.get(id - 1);
	}
	
	@PostMapping("/tasks")
	public void addTask(@RequestBody Task newTask) {
		todoList.add(newTask);
	}
}
