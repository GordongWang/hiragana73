function feval(fn_str, args...)
    return eval(parse(fn_str))(args...)
end
