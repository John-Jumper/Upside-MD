#include "thread_task.h"

using namespace std;

Task::Task(const std::vector<Task*>& dependencies_,
        std::function<int(int)>&&          controller_fcn_,
        std::function<void(int,int,int)>&& subtask_fcn_):
    state(State::WaitingForDependency),
    n_round(0),
    n_subtasks(0),
    n_subtasks_attempted(0),
    n_subtasks_completed(0),
    controller_fcn(controller_fcn_),
    subtask_fcn(subtask_fcn_)
{
    first_controller_is_claimed.clear();

    for(size_t i=0; i<MAX_DEPENDENCIES; ++i)
        dependencies[i] = i<dependencies_.size() ? dependencies_[i] : nullptr;
}


bool Task::attempt_progress_and_return_is_complete(const std::vector<Task>& tasks) {
    using namespace std;

    if(state.load() == State::Complete) return true;

    if(state == State::WaitingForDependency) {
        for(const Task* dep: dependencies) {
            if(!dep) break;
            if(dep->state != State::Complete) return false;
        }

        // Now we need to claim the right to run the serial part or continue if already claimed
        if(first_controller_is_claimed.test_and_set())
            return false; 
        set_n_subtasks_with_controller(0);
        if(state.load()==State::Complete) return true;
    }

    // Now state must be Running or later

    int my_task_num = -1;
    do {
        int my_n_subtasks = -1;  // local copy to avoid reading outside mutex
        int my_n_round    = -1;
        {
            lock_guard<mutex> g(mut);
            if(n_subtasks_attempted < n_subtasks) {
                my_task_num = n_subtasks_attempted++;
                my_n_subtasks = n_subtasks;
                my_n_round = n_round;
            }
        }

        if(my_task_num != -1) {
            subtask_fcn(my_n_round, my_task_num, my_n_subtasks);
            bool do_controller = false;

            // Now report that we finished
            {
                lock_guard<mutex> g(mut);
                ++n_subtasks_completed;

                // last finisher runs the controller
                if(n_subtasks_completed == n_subtasks) {
                    do_controller = true;
                    n_round = my_n_round+1;
                    my_n_round = n_round;
                }
            }

            if(do_controller)
                set_n_subtasks_with_controller(my_n_round);
        }
    } while(my_task_num != -1);  // continue loop if we completed a subtask

    return state.load()==State::Complete;
}

void Task::set_n_subtasks_with_controller(int controller_round) {
    // Takes a lock
    auto n_new_subtasks = controller_fcn(controller_round);
    {
        lock_guard<mutex> g(mut);

        n_subtasks = n_new_subtasks;
        n_subtasks_attempted = 0;
        n_subtasks_completed = 0;
        state.store(n_subtasks ? State::Running : State::Complete);
    }
}


void TaskGraphExecutor::process_graph() {
    // check for shutdown on every iteration
    for(bool all_complete = false; !all_complete && !do_shutdown.load(); ) {
        all_complete = true;

        for(auto& t: tasks)
            all_complete &= t.attempt_progress_and_return_is_complete(tasks);
    }
}

void TaskGraphExecutor::worker_loop() {
    while(!do_shutdown.load()) {
        // spin wait or condition variable wait -- either must use a loop
        // make sure at least one iteration has passed so we don't miss a
        // necessary wait
        do{
            if(!spin_wait_between_graphs.load()) {
                std::unique_lock<std::mutex> g(mut);
                cv.wait(g); // wait for notify
            }
        } while(!main_thread_is_ready.load());

        process_graph();

        ++n_complete_threads;
        while(n_complete_threads.load() != n_threads && !do_shutdown.load());
    }
}

void TaskGraphExecutor::execute_graph() {
    if(1+int(worker_threads.size()) != n_threads)
        throw std::string("Threads have not been started!");

    n_complete_threads.store(0);

    if(spin_wait_between_graphs.load()) {
        main_thread_is_ready.store(true);
    } else {
        // I am a bit naive on this.  I am not sure if I need to take the lock 
        // to modify the condition, but I want to be paranoid and follow the docs
        {
            std::lock_guard<std::mutex> g(mut);
            main_thread_is_ready.store(true);
        }
        cv.notify_all();
    }

    process_graph();

    // wait for everyone to finish graph so that we are safe to modify tasks again
    while(n_complete_threads.load() != n_threads-1 && !do_shutdown.load());
    main_thread_is_ready.store(false);
    ++n_complete_threads;
}


TaskGraphExecutor::TaskGraphExecutor(int n_threads_):
    main_thread_is_ready(false),
    n_complete_threads(0),
    do_shutdown(false),
    spin_wait_between_graphs(true),
    n_threads(n_threads_)
{}

void TaskGraphExecutor::start_threads()
{
    for(int nt=0; nt<n_threads-1; ++nt)
        worker_threads.emplace_back([this](){this->worker_loop();});
}

TaskGraphExecutor::~TaskGraphExecutor() {
    do_shutdown.store(true);
    for(auto& t: worker_threads) t.join();
}
