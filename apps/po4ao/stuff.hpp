class CircularBuffer {
    private:
        size_t num_rows, num_cols;     // Dimensions of each image (rows x columns)
        size_t historySize;            // Total number of images the buffer can hold
        size_t size;                   // Total number of elements in each image (rows * columns)
        float* buffer;                 // Buffer holding the images
        size_t head;                   // Index for the next write position
        size_t tail;                   // Index for the next read position
        bool is_full;                  // Flag to check if buffer is full

    public:
        CircularBuffer(size_t _num_rows, size_t _num_cols, size_t _historySize)
            : num_rows(_num_rows), num_cols(_num_cols), 
              historySize(_historySize), 
              size(_num_rows * _num_cols), 
              head(0), tail(0), is_full(false) {
            
            buffer = new float[size * historySize];
        }

        ~CircularBuffer(){
            delete buffer;
        }
    
        // Adds an image (array of floats) to the buffer (overwrites if full)
        void add(float* image) {
            memcpy(buffer[(head % historySize) * size], image, sizeof(float) * size);
            head = head + 1;
        }

        void get(float* dest, int index, int length){
            if(index < num_elements()){
                int data_index = (head - 1 - index) % historySize;
                memcpy(dest, buffer[data_index * size], sizeof(float) * size);
            }
        }

        void add_eigenimage(Eigen::Map<eigenImage<unsigned short>> image) {
            for (size_t i = 0; i < image.size(); ++i) {
                buffer[head][i] = image(i); // Cast to float if needed
            }
        }

        int num_elements(){
            if( head > historySize){
                return historySize;
            }else{
                return head;
            }
        }

        void reset_head(){
            head = 0;
        }
    
        // Returns the n-th image in the buffer in the order of addition
        std::vector<float> getItem(size_t n) const {
            std::cout << "HOI" << std::endl;
            std::cout << head << std::endl;
            size_t index = head  - n;
            std::cout << "Accessing index " << index << std::endl;
            return buffer[index];
        }

        std::vector<std::vector<float>> getBuffer() const {
            return buffer;
        }
    

};