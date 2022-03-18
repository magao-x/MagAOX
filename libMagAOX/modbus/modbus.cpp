

#include "modbus.hpp"


modbus::modbus( const std::string & host, 
                uint16_t port
              ) : PORT {port}, HOST{host} {}

modbus::modbus(const std::string & host) : HOST{host} {}

modbus::~modbus(void) {
}

void modbus::modbus_set_slave_id(int id) {
    _slaveid = id;
}

bool modbus::modbus_connect() {
    if(HOST == "" || PORT == 0) {
        //std::cout << "Missing Host and Port" << std::endl;
        return false;
    } else {
        //std::cout << "Found Proper Host "<< HOST << " and Port " <<PORT <<std::endl;
    }

    _socket = socket(AF_INET, SOCK_STREAM, 0);
    if(_socket == -1) {
        //std::cout <<"Error Opening Socket" <<std::endl;
        return false;
    } else {
        //std::cout <<"Socket Opened Successfully" << std::endl;
    }

    _server.sin_family = AF_INET;
    _server.sin_addr.s_addr = inet_addr(HOST.c_str());
    _server.sin_port = htons(PORT);

    if (connect(_socket, (struct sockaddr*)&_server, sizeof(_server)) < 0) {
        //std::cout<< "Connection Error" << std::endl;
        return false;
    }

    //std::cout<< "Connected" <<std::endl;
    _connected = true;
    return true;
}

void modbus::modbus_close() {
    close(_socket);
    //std::cout <<"Socket Closed" <<std::endl;
}

void modbus::modbus_build_request(uint8_t *to_send, int address, int func) {
    to_send[0] = (uint8_t) _msg_id >> 8;
    to_send[1] = (uint8_t) (_msg_id & 0x00FF);
    to_send[2] = 0;
    to_send[3] = 0;
    to_send[4] = 0;
    to_send[6] = (uint8_t) _slaveid;
    to_send[7] = (uint8_t) func;
    to_send[8] = (uint8_t) (address >> 8);
    to_send[9] = (uint8_t) (address & 0x00FF);
}

void modbus::modbus_write(int address, int amount, int func, uint16_t *value) {
    if(func == WRITE_COIL || func == WRITE_REG) {
        uint8_t to_send[12];
        modbus_build_request(to_send, address, func);
        to_send[5] = 6;
        to_send[10] = (uint8_t) (value[0] >> 8);
        to_send[11] = (uint8_t) (value[0] & 0x00FF);
        modbus_send(to_send, 12);
    } else if(func == WRITE_REGS){
        uint8_t to_send[13 + 2 * amount];
        modbus_build_request(to_send, address, func);
        to_send[5] = (uint8_t) (5 + 2 * amount);
        to_send[10] = (uint8_t) (amount >> 8);
        to_send[11] = (uint8_t) (amount & 0x00FF);
        to_send[12] = (uint8_t) (2 * amount);
        for(int i = 0; i < amount; i++) {
            to_send[13 + 2 * i] = (uint8_t) (value[i] >> 8);
            to_send[14 + 2 * i] = (uint8_t) (value[i] & 0x00FF);
        }
        modbus_send(to_send, 13 + 2 * amount);
    } else if(func == WRITE_COILS) {
        uint8_t to_send[14 + (amount -1) / 8 ];
        modbus_build_request(to_send, address, func);
        to_send[5] = (uint8_t) (7 + (amount -1) / 8);
        to_send[10] = (uint8_t) (amount >> 8);
        to_send[11] = (uint8_t) (amount >> 8);
        to_send[12] = (uint8_t) ((amount + 7) / 8);
        for(int i = 0; i < amount; i++) {
            to_send[13 + (i - 1) / 8] += (uint8_t) (value[i] << (i % 8));
        }
        modbus_send(to_send, 14 + (amount - 1) / 8);
    }
}

void modbus::modbus_read(int address, int amount, int func){
    uint8_t to_send[12];
    modbus_build_request(to_send, address, func);
    to_send[5] = 6;
    to_send[10] = (uint8_t) (amount >> 8);
    to_send[11] = (uint8_t) (amount & 0x00FF);
    modbus_send(to_send, 12);
}

void modbus::modbus_read_holding_registers(int address, int amount, uint16_t *buffer) {
    if(_connected) {
        if(amount > 65535 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_REGS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_REGS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = ((uint16_t)to_rec[9 + 2 * i]) << 8;
                buffer[i] += (uint16_t) to_rec[10 + 2 * i];
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_read_input_registers(int address, int amount, uint16_t *buffer) {
    if(_connected){
        if(amount > 65535 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_INPUT_REGS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_INPUT_REGS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = ((uint16_t)to_rec[9 + 2 * i]) << 8;
                buffer[i] += (uint16_t) to_rec[10 + 2 * i];
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_read_coils(int address, int amount, bool *buffer) {
    if(_connected) {
        if(amount > 2040 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_COILS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_COILS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = (bool) ((to_rec[9 + i / 8] >> (i % 8)) & 1);
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_read_input_bits(int address, int amount, bool* buffer) {
    if(_connected) {
        if(amount > 2040 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_INPUT_BITS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_INPUT_BITS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = (bool) ((to_rec[9 + i / 8] >> (i % 8)) & 1);
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_write_coil(int address, bool to_write) {
    if(_connected) {
        if(address > 65535) {
            throw modbus_amount_exception();
        }
        int value = to_write * 0xFF00;
        modbus_write(address, 1, WRITE_COIL, (uint16_t *)&value);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_COIL);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_write_register(int address, uint16_t value) {
    if(_connected) {
        if(address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_write(address, 1, WRITE_REG, &value);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_COIL);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_write_coils(int address, int amount, bool *value) {
    if(_connected) {
        if(address > 65535 || amount > 65535) {
            throw modbus_amount_exception();
        }
        uint16_t temp[amount];
        for(int i = 0; i < 4; i++) {
            temp[i] = (uint16_t)value[i];
        }
        modbus_write(address, amount, WRITE_COILS,  temp);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_COILS);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

void modbus::modbus_write_registers(int address, int amount, uint16_t *value) {
    if(_connected) {
        if(address > 65535 || amount > 65535) {
            throw modbus_amount_exception();
        }
        modbus_write(address, amount, WRITE_REGS, value);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_REGS);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}

ssize_t modbus::modbus_send(uint8_t *to_send, int length) {
    _msg_id++;
    return send(_socket, to_send, (size_t)length, 0);
}

ssize_t modbus::modbus_receive(uint8_t *buffer) {
    return recv(_socket, (char *) buffer, MAX_MSG_LENGTH, 0);
}

void modbus::modbus_error_handle(uint8_t *msg, int func) {
    if(msg[7] == func + 0x80) {
        switch(msg[8]){
            case EX_ILLEGAL_FUNCTION:
                throw modbus_illegal_function_exception();
            case EX_ILLEGAL_ADDRESS:
                throw modbus_illegal_address_exception();
            case EX_ILLEGAL_VALUE:
                throw modbus_illegal_data_value_exception();
            case EX_SERVER_FAILURE:
                throw modbus_server_failure_exception();
            case EX_ACKNOWLEDGE:
                throw modbus_acknowledge_exception();
            case EX_SERVER_BUSY:
                throw modbus_server_busy_exception();
            case EX_GATEWAY_PROBLEMP:
            case EX_GATEWYA_PROBLEMF:
                throw modbus_gateway_exception();
            default:
                break;
        }
    }
}
