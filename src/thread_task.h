#ifndef THREAD_TASK_H
#define THREAD_TASK_H

#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

constexpr int MAX_DEPENDENCIES = 10;

struct Task {
    enum class State {
        WaitingForDependency, // not ready to run
        Running,              // not all subtasks are assigned to threads
        Complete};            // work is finished (including finalize_task function) 

    std::atomic<State> state;
    std::array<Task*,MAX_DEPENDENCIES> dependencies; // keep this on the same cache line

    // first invocation of the controller is special since we must decide
    // which thread completed dependencies.  For all other invocations, the last to increment
    // the n_subtasks_completed executes
    std::atomic_flag first_controller_is_claimed;
    
    // mutex is required to read or write the int's below
    std::mutex mut;
    int n_round;
    int n_subtasks;
    int n_subtasks_attempted; // may exceed n_subtasks slightly due to race conditions
    int n_subtasks_completed;

    std::function<int(int)>          controller_fcn; // input is n_round and return is new n_subtasks
    std::function<void(int,int,int)> subtask_fcn;    // input is n_round and subtask and n_subtasks
                        
    Task(const std::vector<Task*>& dependencies_,
            std::function<int(int)>&&          controller_fcn_,
            std::function<void(int,int,int)>&& subtask_fcn_);

    // returns true if node is complete after executions (thread-safe)
    bool attempt_progress_and_return_is_complete(const std::vector<Task>& tasks);  

    void set_n_subtasks_with_controller(int controller_round);
};


struct TaskGraphExecutor {
    protected:
        std::vector<std::thread> worker_threads;
        std::atomic<bool> main_thread_is_ready;
        std::atomic<int> n_complete_threads;
        std::atomic<bool> do_shutdown;

        std::mutex mut;
        std::condition_variable cv;
        std::atomic<bool> spin_wait_between_graphs;

        void process_graph();
        void worker_loop();

    public:
        std::vector<Task> tasks;
        const int n_threads; // includes "main" thread

        TaskGraphExecutor(int n_threads_);
        void start_threads();
        void execute_graph();
        virtual ~TaskGraphExecutor();
};

#endif
