#include "task_graph.h"

using namespace std;

Task::Task(const std::string& name_, 
        std::function<int(int)>&&          controller_fcn_,
        std::function<void(int,int,int)>&& subtask_fcn_):
    n_threads_needed(0),
    complete(false),
    n_dependencies_unsatisfied(0),
    n_subtasks_completed(0),

    n_round(-1),
    n_subtasks(0),

    controller_fcn(std::move(controller_fcn_)),
    subtask_fcn(std::move(subtask_fcn_)),

    name(name_)
{}


Task::Task(Task&& o):
    n_threads_needed(o.n_threads_needed.load()),
    complete(o.complete.load()),
    n_dependencies_unsatisfied(o.n_dependencies_unsatisfied.load()),
    n_subtasks_completed(o.n_subtasks_completed.load()),

    n_round(o.n_round),
    n_subtasks(o.n_subtasks),

    controller_fcn(std::move(o.controller_fcn)),
    subtask_fcn(std::move(o.subtask_fcn)),

    consumers(std::move(o.consumers)),
    name(std::move(o.name))
{}

void TaskGraphExecutor::process_graph() {
    // printf("starting graph\n");
    for(bool all_complete = false; !all_complete && !do_shutdown.load(); ) {
        // printf("new iteration\n");
        all_complete = true;
        for(auto& t: tasks) {
            while(t.n_threads_needed.load() > 0) {
                int task_num = --t.n_threads_needed;
                if(task_num < 0) break;

                bool need_to_do_controller = t.n_round == -1;

                if(!need_to_do_controller) {
                    t.subtask_fcn(t.n_round, task_num, t.n_subtasks);
                    need_to_do_controller = t.n_subtasks == ++t.n_subtasks_completed;
                }

                if(need_to_do_controller) {
                    // printf("CONTROLLER\n");
                    if(t.n_round == -1) t.n_round++;
                    auto n_new_subtasks = t.controller_fcn(t.n_round);
                    if(!n_new_subtasks){
                        t.complete.store(true);
                        for(int i: t.consumers) {
                            int dep_remaining = --tasks[i].n_dependencies_unsatisfied;
                            if(!dep_remaining) tasks[i].n_threads_needed.store(1);
                        }
                    }

                    ++t.n_round;
                    t.n_subtasks = n_new_subtasks;
                    t.n_subtasks_completed.store(0);
                    t.n_threads_needed.store(n_new_subtasks);
                }
            }
            all_complete &= t.complete.load();
            // printf("%-42s %i %i %s\n",
            //         t.name.c_str(), t.n_threads_needed.load(),
            //         t.n_dependencies_unsatisfied.load(),
            //         t.complete.load() ? "complete" : "");
        }
    } 
}


void TaskGraphExecutor::worker_loop() {
    while(!do_shutdown.load()) {
        ++n_workers_ready;

        // spin wait or condition variable wait -- either must use a loop
        // make sure at least one iteration has passed so we don't miss a
        // necessary wait
        do{
            if(!spin_wait_between_graphs.load()) {
                std::unique_lock<std::mutex> g(mut);
                cv.wait(g); // wait for notify
            }
        } while(!main_ready.load());

        process_graph();
        ++n_workers_complete;

        while(!main_complete.load() && !do_shutdown.load());
    }
}


void TaskGraphExecutor::execute() {
    if(n_workers!=int(worker_threads.size())) start_threads();
    while(n_workers_ready.load() != n_workers && !do_shutdown.load());

    // let's reset all the tasks now
    for(auto& t: tasks) {
        t.complete.store(false);
        t.n_round = -1;
        for(int i: t.consumers)
            ++tasks[i].n_dependencies_unsatisfied;
    }
    
    // Any tasks that have no dependencies should be marked as needing threads
    for(auto& t: tasks)
        if(!t.n_dependencies_unsatisfied.load()) t.n_threads_needed.store(1);

    main_complete.store(false);
    n_workers_complete.store(0);

    if(spin_wait_between_graphs.load()) {
        main_ready.store(true);
    } else {
        {   // for a condition_variable, this must update must be locked
            std::lock_guard<std::mutex> g(mut);
            main_ready.store(true);
        }
        cv.notify_all();
    }

    process_graph();

    // wait for everyone to finish graph so that we are safe to modify tasks again
    while(n_workers_complete.load() != n_workers && !do_shutdown.load());
    n_workers_ready.store(0);
    main_ready   .store(false);
    main_complete.store(true);
}


TaskGraphExecutor::TaskGraphExecutor(int n_workers_):
    main_ready(false),
    main_complete(true),
    n_workers_ready(0),
    n_workers_complete(n_workers_),
    do_shutdown(false),
    
    spin_wait_between_graphs(true),
    n_workers(n_workers_)
{}


void TaskGraphExecutor::start_threads()
{
    for(int nt=0; nt<n_workers; ++nt)
        worker_threads.emplace_back([this](){this->worker_loop();});
}


TaskGraphExecutor::~TaskGraphExecutor() {
    do_shutdown.store(true);
    for(auto& t: worker_threads) t.join();
}
