local ReQU, parent = torch.class('nn.ReQU', 'nn.Module')

function ReQU:__init()
    parent.__init(self)
end

function ReQU:updateOutput(input)
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:pow(2)
    self.output:cmul(torch.lt(input, 0):double())
    return self.output
end

function ReQU:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(input)
    self.gradInput:mul(2)
    self.gradInput:cmul(gradOutput)
    self.gradInput:cmul(torch.lt(input, 0):double())
    return self.gradInput
end
