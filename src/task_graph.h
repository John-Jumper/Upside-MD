#ifndef THREAD_TASK_H
#define THREAD_TASK_H

#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <string>

struct Task {
    std::atomic<int>  n_threads_needed;
    std::atomic<bool> complete;
    std::atomic<int>  n_dependencies_unsatisfied;
    std::atomic<int>  n_subtasks_completed;

    int n_round;
    int n_subtasks;

    //! input is n_round and return is new n_subtasks
    std::function<int(int)>          controller_fcn;

    //! input is n_round and subtask and n_subtasks
    std::function<void(int,int,int)> subtask_fcn;
    
    //! Vector of every task index that depends on this task
    std::vector<size_t> consumers;
    std::string name; // primarily for debugging
                        
    Task(const std::string& name_, 
            std::function<int(int)>&&          controller_fcn_,
            std::function<void(int,int,int)>&& subtask_fcn_);

    Task(Task&& o);
};



struct TaskGraphExecutor {
    protected:
        std::vector<std::thread> worker_threads;

        // FIXME I may want these variables on different cache lines to avoid
        // lots of invalidations
        std::atomic<bool> main_ready;
        std::atomic<bool> main_complete;
        std::atomic<int>  n_workers_ready;
        std::atomic<int>  n_workers_complete;
        std::atomic<bool> do_shutdown;

        std::mutex mut;
        std::condition_variable cv;
        std::atomic<bool> spin_wait_between_graphs;

        void process_graph();
        void start_threads();
        void worker_loop();
        void set_n_subtasks_with_controller(int task_idx, int controller_round);

    public:
        std::vector<Task> tasks;
        const int n_workers; // does not include main thread

        TaskGraphExecutor(int n_workers_);
        void execute();
        virtual ~TaskGraphExecutor();
};

#endif