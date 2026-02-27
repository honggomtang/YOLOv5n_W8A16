`timescale 1ns / 1ps

module tb_conv_acc;

    localparam CLK_PERIOD = 10;
    localparam NUM_CLUSTERS = 8;
    localparam NUM_PE = 32;
    localparam MAX_W_LINE = 3072;

    reg aclk, aresetn;
    reg [31:0] s_axis_tdata;
    reg s_axis_tvalid, s_axis_tlast;
    wire s_axis_tready;
    wire [31:0] m_axis_tdata;
    wire m_axis_tvalid, m_axis_tlast;
    reg m_axis_tready;

    reg [3:0]  kernel_size;
    reg [31:0] target_ic;
    reg [15:0] img_width;
    reg [31:0] multiplier;
    reg [3:0]  stride_val;
    reg [6:0]  act_start;
    reg        start_load;

    conv_acc_top uut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),
        .kernel_size(kernel_size),
        .target_ic(target_ic),
        .img_width(img_width),
        .multiplier(multiplier),
        .stride_val(stride_val),
        .act_start(act_start),
        .start_load(start_load),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tready(m_axis_tready)
    );

    initial aclk = 0;
    always #(CLK_PERIOD/2) aclk = ~aclk;

    task send_bias;
        integer i;
        begin
            for (i = 0; i < 32; i = i + 1) begin
                @(posedge aclk);
                s_axis_tdata = (i < 16) ? 32'd100 : 32'd0;
                s_axis_tvalid = 1;
                s_axis_tlast = (i == 31);
                while (!s_axis_tready) @(posedge aclk);
            end
            @(posedge aclk);
            s_axis_tvalid = 0;
            s_axis_tlast = 0;
        end
    endtask

    task send_weight;
        input integer n_words;
        integer i;
        begin
            for (i = 0; i < n_words; i = i + 1) begin
                @(posedge aclk);
                s_axis_tdata = (i % 4 == 0) ? 32'h01010101 : 32'd0;
                s_axis_tvalid = 1;
                s_axis_tlast = (i == n_words - 1);
                while (!s_axis_tready) @(posedge aclk);
            end
            @(posedge aclk);
            s_axis_tvalid = 0;
            s_axis_tlast = 0;
        end
    endtask

    task send_weight_pos_dependent;
        input integer n_words;
        integer i, c;
        begin
            for (i = 0; i < n_words; i = i + 1) begin
                c = i % 8;
                @(posedge aclk);
                s_axis_tdata = (i == 0) ? 32'h02010101 :
                    ((c == 0 || c == 4) ? 32'h01010101 : 32'd0);
                s_axis_tvalid = 1;
                s_axis_tlast = (i == n_words - 1);
                while (!s_axis_tready) @(posedge aclk);
            end
            @(posedge aclk);
            s_axis_tvalid = 0;
            s_axis_tlast = 0;
        end
    endtask

    task send_act;
        input integer n_words;
        integer i;
        begin
            for (i = 0; i < n_words; i = i + 1) begin
                @(posedge aclk);
                s_axis_tdata = 32'h00010001;
                s_axis_tvalid = 1;
                s_axis_tlast = (i == n_words - 1);
                while (!s_axis_tready) @(posedge aclk);
            end
            @(posedge aclk);
            s_axis_tvalid = 0;
            s_axis_tlast = 0;
        end
    endtask

    integer rx_word_count;
    integer rx_last_seen;
    integer rx_last_cycle;
    reg [31:0] first_pixel [0:15];
    integer first_pixel_captured;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            rx_word_count <= 0;
            rx_last_seen <= 0;
            rx_last_cycle <= 0;
            first_pixel_captured <= 0;
        end else if (m_axis_tvalid && m_axis_tready) begin
            if (rx_word_count < 16)
                first_pixel[rx_word_count] <= m_axis_tdata;
            if (rx_word_count == 15)
                first_pixel_captured <= 1;
            rx_word_count <= rx_word_count + 1;
            if (m_axis_tlast) begin
                rx_last_seen <= 1;
                rx_last_cycle <= rx_word_count + 1;
            end
        end
    end

    integer expected_words;
    integer row_idx;
    integer act_words;
    integer weight_words;

    initial begin
        $display("========================================");
        $display("  conv_acc_top TB - L1 1 row test");
        $display("========================================");

        aresetn = 0;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tlast = 0;
        m_axis_tready = 1;
        kernel_size = 4'd3;
        target_ic = 32'd16;
        img_width = 16'd322;
        multiplier = 32'd131072;
        stride_val = 4'd2;
        act_start = 7'd0;
        start_load = 0;

        #(CLK_PERIOD*5);
        aresetn = 1;
        #(CLK_PERIOD*3);

        expected_words = 160 * 16;
        act_words = 3 * MAX_W_LINE;
        weight_words = 16 * 3 * 3 * NUM_CLUSTERS;

        $display("[TB] L1 params: k=3, ic=16, img_w=322, stride=2");
        $display("[TB] Expected output: %0d words/row, tlast on last word", expected_words);

        $display("[TB] Row 0: start_load=1, Bias+Weight+Act tx");
        start_load = 1;
        rx_word_count = 0;
        rx_last_seen = 0;

        @(posedge aclk);
        send_bias;
        $display("[TB] Bias 32 words tx done");

        send_weight(weight_words);
        $display("[TB] Weight %0d words tx done", weight_words);

        start_load = 0;

        send_act(act_words);
        $display("[TB] Act %0d words tx done", act_words);

        begin : wait_rx0
            integer i;
            for (i = 0; i < 50000; i = i + 1) begin
                @(posedge aclk);
                if (rx_last_seen) disable wait_rx0;
            end
        end

        $display("");
        $display("--- Row 0 result ---");
        $display("RX word count: %0d (expected: %0d)", rx_word_count, expected_words);
        $display("tlast seen: %s (at word #%0d)", rx_last_seen ? "YES" : "NO", rx_last_cycle);
        $display("First pixel (16 words, ch0=low16 of w0):");
        $display("  w0=0x%08X w1=0x%08X w2=0x%08X w3=0x%08X",
            first_pixel[0], first_pixel[1], first_pixel[2], first_pixel[3]);
        $display("  w4=0x%08X w5=0x%08X w6=0x%08X w7=0x%08X",
            first_pixel[4], first_pixel[5], first_pixel[6], first_pixel[7]);
        if (rx_word_count >= 1 && (first_pixel[0] & 16'hFFFF) == 16'd1)
            $display("FAIL: first ch = 0x0001 (out_valid 1clk early capture bug)");
        else if (rx_word_count >= 1 && (first_pixel[0] & 16'hFFFF) == 16'hFFFF)
            $display("WARN: first ch = 0xFFFF (possible pipeline/clear bug)");
        else if (rx_word_count >= 1 && (first_pixel[0] & 16'hFFFF) != 16'd488)
            $display("FAIL: first ch = %0d (0x%04X), expected 488 (0x01E8)", first_pixel[0] & 16'hFFFF, first_pixel[0] & 16'hFFFF);
        else if (rx_word_count >= 1)
            $display("PASS: first pixel ch0 = 488 (0x01E8) golden OK");
        if (rx_word_count == expected_words && rx_last_seen)
            $display("PASS: Row 0 output OK");
        else if (rx_word_count != expected_words)
            $display("FAIL: word count mismatch (diff: %0d)", rx_word_count - expected_words);
        else if (!rx_last_seen)
            $display("FAIL: tlast not seen - likely cause of DMA RX hang");
        $display("");

        $display("[TB] Row 1: start_load=0, Act only (row continue)");
        rx_word_count = 0;
        rx_last_seen = 0;

        repeat (100) @(posedge aclk);

        send_act(act_words);
        $display("[TB] Act %0d words tx done", act_words);

        begin : wait_rx1
            integer j;
            for (j = 0; j < 50000; j = j + 1) begin
                @(posedge aclk);
                if (rx_last_seen) disable wait_rx1;
            end
        end

        $display("");
        $display("--- Row 1 result ---");
        $display("RX word count: %0d (expected: %0d)", rx_word_count, expected_words);
        $display("tlast seen: %s (at word #%0d)", rx_last_seen ? "YES" : "NO", rx_last_cycle);
        if (rx_word_count == expected_words && rx_last_seen)
            $display("PASS: Row 1 output OK");
        else if (!rx_last_seen)
            $display("FAIL: Row 1 tlast not seen (row-continue mode issue?)");
        $display("");

        $display("[TB] Position-dep weight test: expect ch0=501 (rd_w_addr indexing)");
        aresetn = 0;
        #(CLK_PERIOD*3);
        aresetn = 1;
        #(CLK_PERIOD*3);
        start_load = 1;
        rx_word_count = 0;
        rx_last_seen = 0;
        @(posedge aclk);
        send_bias;
        send_weight_pos_dependent(weight_words);
        start_load = 0;
        send_act(act_words);
        begin : wait_rx2
            integer k;
            for (k = 0; k < 50000; k = k + 1) begin
                @(posedge aclk);
                if (rx_last_seen) disable wait_rx2;
            end
        end
        if (rx_word_count >= 1 && (first_pixel[0] & 16'hFFFF) == 16'd501)
            $display("PASS: pos-dep ch0 = 501 (weight indexing OK)");
        else if (rx_word_count >= 1)
            $display("FAIL: pos-dep ch0 = %0d, expected 501 (possible rd_w_addr bug)", first_pixel[0] & 16'hFFFF);
        $display("");

        $display("========================================");
        $display("  Simulation done");
        $display("========================================");
        #(CLK_PERIOD*10);
        $finish;
    end

endmodule
