#include "task_graph.h"

using namespace std;


Task::Task(const string& name_, const int cost_,
        function<int(int)>&&          controller_fcn_,
        function<void(int,int,int)>&& subtask_fcn_):
    n_dep_unsatisfied(0),

    n_round(-1),
    n_subtasks(1),
    n_subtasks_completed(0),
    priority(cost_),

    controller_fcn(move(controller_fcn_)),
    subtask_fcn(move(subtask_fcn_)),

    name(name_),
    cost(cost_)
{}


Task::Task(Task&& o):
    n_dep_unsatisfied(o.n_dep_unsatisfied.load()),

    n_round(o.n_round),
    n_subtasks(o.n_subtasks),
    n_subtasks_completed(o.n_subtasks_completed),
    priority(o.priority),

    controller_fcn(move(o.controller_fcn)),
    subtask_fcn   (move(o.subtask_fcn)),
    consumers     (move(o.consumers)),

    name(o.name),
    cost(o.cost)
{}

//! pre-decrement with relaxed memory ordering
inline int relaxed_dec(std::atomic<int>& i) {
    return i.fetch_sub(1, memory_order_relaxed)-1;
}

void TaskGraphExecutor::process_graph() {
    vector<RunTask> runtasks_to_add;
    const auto relax = memory_order_relaxed; // avoid typing

    while(true) {
        // A simple loop to reduce stress on the mutex
        // when no work is available

        while(pq_threads_needed.load(relax)<=0);

        int thread_num = relaxed_dec(pq_threads_needed);

        // now we are likely (though not certain) to get work
        int subtask_idx = -1;
        int task_idx = -1;
        if(thread_num>=0) {
            // now we should check
            lock_guard<mutex> g(pq_mutex);
            if(n_incomplete_tasks.load(relax)<=0) return;

            if(pq.size()) {  // there should be work, but could be a race condition
                const auto& top_rt = pq.top();
                task_idx = top_rt.task_idx;
                subtask_idx = --top_rt.n_threads_needed;
                if(!subtask_idx) pq.pop(); // this job no longer needs threads
            }
        }

        // if we got this far without getting a task_idx assigned, then we have
        // decremented pq_threads_needed without doing any work, throwing off
        // the count.  We will increment the counter to return our overdraft.
        if(task_idx == -1) pq_threads_needed.fetch_add(1, relax);

        while(task_idx != -1) {
            Task& t = tasks[task_idx];

            int my_n_round, my_n_subtasks;
            {
                lock_guard<mutex> g(t.mut);
                my_n_round = t.n_round;
                my_n_subtasks = t.n_subtasks;
            }

            bool do_controller = my_n_round==-1;
            if(!do_controller) {
                t.subtask_fcn(my_n_round, subtask_idx, my_n_subtasks);
                {
                    lock_guard<mutex> g(t.mut);
                    do_controller = ++t.n_subtasks_completed == my_n_subtasks;
                }
            }

            if(do_controller) {
                // At the controller stage, only as single thread can be executing
                // so we don't have to take the task mutex
                my_n_round = ++t.n_round;
                auto n_new_subtasks = t.controller_fcn(my_n_round);
                int n_new_threads_needed = 0;

                if(n_new_subtasks) {
                    t.n_subtasks = n_new_subtasks;
                    t.n_subtasks_completed = 0;
                    runtasks_to_add.emplace_back(RunTask{
                            t.priority, uint16_t(task_idx), uint16_t(n_new_subtasks)});
                    n_new_threads_needed += n_new_subtasks;
                } else {
                    int tasks_remaining = relaxed_dec(n_incomplete_tasks);
                    // Make sure everyone gets a chance to see it
                    if(!tasks_remaining) pq_threads_needed.fetch_add(1+n_workers, relax);

                    // Handle consumers outside the pq_mutex for efficiency
                    for(int i: t.consumers) {
                        int dep_remaining = relaxed_dec(tasks[i].n_dep_unsatisfied);
                        if(!dep_remaining) {
                            runtasks_to_add.emplace_back(RunTask{
                                    tasks[i].priority, uint16_t(i), uint16_t(1)});
                            n_new_threads_needed++;
                        }
                    }
                }

                // Now we take the pq_mutex to add the new run tasks
                if(n_new_threads_needed) {
                    {
                        lock_guard<mutex> g(pq_mutex);
                        for(auto& rt: runtasks_to_add) pq.push(rt);

                        // pull off a task while this thread the mutex
                        const auto& top_rt = pq.top();
                        task_idx = top_rt.task_idx;
                        subtask_idx = --top_rt.n_threads_needed;
                        if(!subtask_idx) pq.pop(); // this job no longer needs threads
                    }
                    pq_threads_needed.fetch_add(n_new_threads_needed-1, relax);
                } else {
                    task_idx = -1;
                }
                runtasks_to_add.clear();

            }  // do_controller
        } // task_idx != -1
    } // infinite loop
}


void TaskGraphExecutor::worker_loop() {
    while(!do_shutdown.load()) {
        ++n_workers_ready;

        // spin wait or condition variable wait -- either must use a loop
        // make sure at least one iteration has passed so we don't miss a
        // necessary wait
        do{
            if(!spin_wait_between_graphs.load()) {
                unique_lock<mutex> g(mut);
                cv.wait(g); // wait for notify
            }
        } while(!main_ready.load() && !do_shutdown);

        process_graph();
        ++n_workers_complete;

        while(!main_complete.load() && !do_shutdown.load());
    }
}


void TaskGraphExecutor::execute() {
    const auto mor = memory_order_relaxed;

    if(n_workers!=int(worker_threads.size())) start_threads();
    while(n_workers_ready.load() != n_workers && !do_shutdown.load());

    // let's reset all the tasks now
    for(auto& t: tasks) {
        t.n_round = -1;
        t.n_dep_unsatisfied.store(0, mor);
    }

    for(auto& t: tasks)
        for(int i: t.consumers)
            tasks[i].n_dep_unsatisfied.fetch_add(1, mor);

    // Any tasks that have no dependencies should be marked as needing threads
    pq_threads_needed.store(0);
    for(size_t i=0; i<tasks.size(); ++i)
        if(!tasks[i].n_dep_unsatisfied.load(mor)) {
            pq_threads_needed.fetch_add(1,mor);
            pq.emplace(RunTask{
                    tasks[i].priority, uint16_t(i), uint16_t(1)});
        }

    n_incomplete_tasks.store(int(tasks.size()));
    main_complete.store(false);
    n_workers_complete.store(0);

    if(spin_wait_between_graphs.load()) {
        main_ready.store(true);
    } else {
        {   // for a condition_variable, this must update must be locked
            lock_guard<mutex> g(mut);
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
    n_incomplete_tasks.store(0);
    pq_threads_needed.store(1+n_workers);
    do_shutdown.store(true);
    for(auto& t: worker_threads) t.join();
}
